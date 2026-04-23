import asyncio
import io
import logging
import os
import re
import time
import traceback

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, StoppingCriteria

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Regex ---
SYSTEM_PROMPT = (
    "You are an expert UI element locator. Given a GUI image and a user's element description, "
    "provide the coordinates of the specified element as a single (x,y) point. "
    "The image resolution is height {height} and width {width}. For elements with area, return the center point.\n\n"
    "Output the coordinate pair exactly:\n(x,y)"
)

COORD_PATTERN = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}

def extract_coordinates(raw_string: str) -> tuple[int, int]:
    if match := COORD_PATTERN.search(raw_string):
        return round(float(match.group(1))), round(float(match.group(2)))
    logger.warning("no coordinates found in model output: %s", raw_string)
    return 0, 0

# --- App Initialization ---
app = FastAPI()

# --- Model & Processor Loading ---
MODEL_PATH = os.environ.get("MODEL_PATH", "HelloKKMe/GTA1-7B")
MAX_NEW_TOKENS = 32

model_kwargs = {
    "pretrained_model_name_or_path": MODEL_PATH,
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda:0",
    "local_files_only": True,
    "low_cpu_mem_usage": True,
}

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs, attn_implementation="flash_attention_2")
    logger.info("model loaded with flash_attention_2")
except Exception:
    logger.warning("flash_attention_2 not available, falling back to sdpa")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs, attn_implementation="sdpa")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=3136,
    # max_pixels=1024 * 576,
    max_pixels=1280 * 720,
    local_files_only=True,
)

# Cache Image Processor settings
IMG_PROC = processor.image_processor
RESIZE_FACTOR = IMG_PROC.patch_size * IMG_PROC.merge_size
MIN_PIXELS = IMG_PROC.size.shortest_edge
MAX_PIXELS = IMG_PROC.size.longest_edge

# --- Concurrency Control ---
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

class TimingHook(StoppingCriteria):
    """Records a synced GPU timestamp after each generated token."""
    def __init__(self):
        self.ts = []
    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize()
        self.ts.append(time.perf_counter())
        return False


def run_inference(inputs):
    """Blocking inference function to be run in a separate thread."""
    t0 = time.perf_counter()
    inputs = inputs.to(model.device)
    torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t0) * 1000

    hook = TimingHook()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
            stopping_criteria=[hook],
        )
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t1) * 1000

    # Decompose: ts[0] = prefill + first decode token; ts[i]-ts[i-1] = decode of (i+1)th token
    ts = hook.ts
    if ts:
        prefill_plus_first_ms = (ts[0] - t1) * 1000
        per_token_ms = [(ts[i] - ts[i - 1]) * 1000 for i in range(1, len(ts))]
    else:
        prefill_plus_first_ms = gen_ms
        per_token_ms = []
    return out, h2d_ms, gen_ms, prefill_plus_first_ms, per_token_ms


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/process/")
async def process(instruction: str = Form(...), image_file: UploadFile = File(...)):
    # Fail fast if server is busy
    if inference_semaphore.locked():
        return JSONResponse(status_code=503, content={"error": "server busy, retry later"})

    async with inference_semaphore:
        start_time = time.perf_counter()
        try:
            if image_file.content_type and image_file.content_type not in ALLOWED_CONTENT_TYPES:
                return JSONResponse(status_code=400, content={"error": "unsupported image format"})

            t = time.perf_counter()
            image_bytes = await image_file.read()
            read_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.load()
            except UnidentifiedImageError:
                return JSONResponse(status_code=400, content={"error": "invalid image file"})
            decode_ms = (time.perf_counter() - t) * 1000

            width, height = image.width, image.height

            t = time.perf_counter()
            resized_height, resized_width = smart_resize(
                height, width, factor=RESIZE_FACTOR,
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
            )
            resized_image = image.resize((resized_width, resized_height))
            scale_x, scale_y = width / resized_width, height / resized_height
            resize_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)},
                {"role": "user", "content": [{"type": "image", "image": resized_image}, {"type": "text", "text": instruction}]}
            ]
            template_t = time.perf_counter()
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            template_ms = (time.perf_counter() - template_t) * 1000

            vision_t = time.perf_counter()
            image_inputs, video_inputs = process_vision_info(messages)
            vision_ms = (time.perf_counter() - vision_t) * 1000

            proc_t = time.perf_counter()
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            proc_ms = (time.perf_counter() - proc_t) * 1000
            preprocess_ms = (time.perf_counter() - t) * 1000

            input_len = inputs.input_ids.shape[1]

            output_ids_batch, h2d_ms, gen_ms, prefill_first_ms, per_token_ms = await asyncio.to_thread(run_inference, inputs)

            t = time.perf_counter()
            generated_ids = output_ids_batch[:, input_len:]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_x, pred_y = extract_coordinates(output_text)
            pred_x, pred_y = pred_x * scale_x, pred_y * scale_y
            post_ms = (time.perf_counter() - t) * 1000

            gen_tokens = generated_ids.shape[1]
            total_ms = (time.perf_counter() - start_time) * 1000

            if per_token_ms:
                avg_dec = sum(per_token_ms) / len(per_token_ms)
                min_dec = min(per_token_ms)
                max_dec = max(per_token_ms)
            else:
                avg_dec = min_dec = max_dec = 0.0
            est_prefill_ms = max(prefill_first_ms - avg_dec, 0.0)

            logger.info(
                "TIMING total=%.1fms | read=%.1f decode=%.1f resize=%.1f preprocess=%.1f(template=%.1f vision=%.1f proc=%.1f) "
                "h2d=%.1f generate=%.1f(prefill~%.1f decode_avg=%.1f min=%.1f max=%.1f n=%d) post=%.1f "
                "| in_tok=%d gen_tok=%d | %dx%d->%dx%d | %s",
                total_ms, read_ms, decode_ms, resize_ms,
                preprocess_ms, template_ms, vision_ms, proc_ms,
                h2d_ms, gen_ms, est_prefill_ms, avg_dec, min_dec, max_dec, len(per_token_ms),
                post_ms, input_len, gen_tokens, width, height, resized_width, resized_height, instruction,
            )
            logger.info("TIMING per_token_ms=%s", ["%.1f" % x for x in per_token_ms])
            return {"x": pred_x, "y": pred_y}

        except Exception as e:
            logger.error("inference failed in %.3fs: %s\n%s", time.time() - start_time, e, traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": "internal server error"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)