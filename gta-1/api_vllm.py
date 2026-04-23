import time
SCRIPT_START = time.perf_counter()

import io
import logging
import os
import re
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from qwen_vl_utils import smart_resize
from transformers import AutoProcessor
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Quiet vLLM's per-request logs
logging.getLogger("vllm").setLevel(logging.WARNING)
logger.info("STARTUP imports loaded in %.2fs", time.perf_counter() - SCRIPT_START)

SYSTEM_PROMPT = (
    "You are an expert UI element locator. Given a GUI image and a user's element description, "
    "provide the coordinates of the specified element as a single (x,y) point. "
    "The image resolution is height {height} and width {width}. For elements with area, return the center point.\n\n"
    "Output the coordinate pair exactly:\n(x,y)"
)

COORD_PATTERN = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


def extract_coordinates(raw: str) -> tuple[int, int]:
    if m := COORD_PATTERN.search(raw):
        return round(float(m.group(1))), round(float(m.group(2)))
    logger.warning("no coordinates found in model output: %s", raw)
    return 0, 0


# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "HelloKKMe/GTA1-7B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
# MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1920 * 1080)))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(3840 * 2160)))
MIN_PIXELS = int(os.environ.get("MIN_PIXELS", "3136"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "12288"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
ENABLE_PREFIX_CACHE = os.environ.get("ENABLE_PREFIX_CACHE", "1").lower() in ("1", "true", "yes", "on")
ENABLE_MM_CACHE = os.environ.get("ENABLE_MM_CACHE", "1").lower() in ("1", "true", "yes", "on")
logger.info("Engine config: max_pixels=%d max_model_len=%d prefix_cache=%s mm_cache=%s",
            MAX_PIXELS, MAX_MODEL_LEN, ENABLE_PREFIX_CACHE, ENABLE_MM_CACHE)

# --- vLLM engine ---
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM_UTIL,
    mm_processor_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
    limit_mm_per_prompt={"image": 1},
    enable_prefix_caching=ENABLE_PREFIX_CACHE,   # cache shared prefix KV across requests
    mm_processor_cache_gb=(4 if ENABLE_MM_CACHE else 0),  # cache vision encoder outputs
    enforce_eager=False,                          # enable CUDA graphs for decode
)
_t = time.perf_counter()
engine = AsyncLLMEngine.from_engine_args(engine_args)
logger.info("STARTUP engine ready in %.2fs", time.perf_counter() - _t)

# Processor only for chat-template formatting (vLLM does its own image processing)
_t = time.perf_counter()
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    local_files_only=True,
)
RESIZE_FACTOR = processor.image_processor.patch_size * processor.image_processor.merge_size
logger.info("STARTUP processor ready in %.2fs", time.perf_counter() - _t)

SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)


@asynccontextmanager
async def lifespan(app):
    elapsed = time.perf_counter() - SCRIPT_START
    bar = "=" * 60
    logger.info(bar)
    logger.info("STARTUP COMPLETE — server ready after %.2fs from script start", elapsed)
    logger.info(bar)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/process/")
async def process(instruction: str = Form(...), image_file: UploadFile = File(...)):
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

        # Compute the dimensions vLLM will use internally so we can:
        #  1) tell the model the right resolution in SYSTEM_PROMPT
        #  2) scale the model's output coords back to the original image
        t = time.perf_counter()
        resized_height, resized_width = smart_resize(
            height, width, factor=RESIZE_FACTOR,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        scale_x, scale_y = width / resized_width, height / resized_height
        resize_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        preprocess_ms = (time.perf_counter() - t) * 1000

        # Submit to vLLM and track time-to-first-token ourselves
        gen_t = time.perf_counter()
        request_id = uuid.uuid4().hex
        results = engine.generate(
            prompt={"prompt": prompt_text, "multi_modal_data": {"image": image}},
            sampling_params=SAMPLING,
            request_id=request_id,
        )
        final_output = None
        ttft_ms = None
        async for output in results:
            if ttft_ms is None and output.outputs and len(output.outputs[0].token_ids) > 0:
                ttft_ms = (time.perf_counter() - gen_t) * 1000
            final_output = output
        gen_ms = (time.perf_counter() - gen_t) * 1000

        # Estimate prefill / decode split from total + ttft
        n_tok = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0
        if ttft_ms and n_tok > 1:
            decode_avg_ms = (gen_ms - ttft_ms) / (n_tok - 1)
            prefill_ms = max(ttft_ms - decode_avg_ms, 0.0)
        else:
            decode_avg_ms = 0.0
            prefill_ms = ttft_ms or 0.0
        queued_ms = 0.0

        t = time.perf_counter()
        output_text = final_output.outputs[0].text
        gen_tokens = len(final_output.outputs[0].token_ids)
        pred_x, pred_y = extract_coordinates(output_text)
        pred_x, pred_y = pred_x * scale_x, pred_y * scale_y
        post_ms = (time.perf_counter() - t) * 1000

        total_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "TIMING total=%.1fms | read=%.1f decode=%.1f resize=%.1f preprocess=%.1f "
            "generate=%.1f(queued=%.1f prefill~%.1f decode_avg=%.1f) post=%.1f "
            "| gen_tok=%d | %dx%d->%dx%d | %s -> (%.1f, %.1f)",
            total_ms, read_ms, decode_ms, resize_ms, preprocess_ms,
            gen_ms, queued_ms, prefill_ms, decode_avg_ms, post_ms,
            gen_tokens, width, height, resized_width, resized_height, instruction, pred_x, pred_y,
        )
        return {"x": pred_x, "y": pred_y}

    except Exception as e:
        logger.error("inference failed in %.3fs: %s\n%s",
                     time.perf_counter() - start_time, e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
