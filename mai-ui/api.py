from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import re
import io
import json
import time
import asyncio
import logging
import traceback
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- System Prompt and Coordinate Extraction ---
SYSTEM_PROMPT = """
You are a GUI grounding agent.
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
## Output Format
Return a json object with a [x,y] format coordinate within <answer></answer> XML tags:
<answer>
{"coordinate": [x,y]}
</answer>
""".strip()

SCALE_FACTOR = 999

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_coordinates(raw_string):
    try:
        match = ANSWER_PATTERN.search(raw_string)
        if not match:
            return 0, 0
        answer_text = match.group(1).strip()
        answer_json = json.loads(answer_text)
        coordinates = answer_json.get("coordinate", [])
        if len(coordinates) == 2:
            return coordinates[0] / SCALE_FACTOR, coordinates[1] / SCALE_FACTOR
        return 0, 0
    except Exception:
        return 0, 0


# --- FastAPI App Initialization ---
app = FastAPI()

# --- Model and Processor Loading (on startup) ---
model_path = "Tongyi-MAI/MAI-UI-8B"
max_new_tokens = 128

try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        local_files_only=True,
    )
    logger.info("model loaded with flash_attention_2")
except Exception:
    logger.warning("flash_attention_2 not available, falling back to sdpa")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        local_files_only=True,
    )
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=3136,
    max_pixels=1920 * 1080,
    local_files_only=True,
)

model = torch.compile(model)
logger.info("model compiled with torch.compile")

resize_factor = processor.image_processor.patch_size * processor.image_processor.merge_size
resize_min_pixels = processor.image_processor.min_pixels
resize_max_pixels = processor.image_processor.max_pixels

MAX_CONCURRENT_REQUESTS = 10
inference_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def run_inference(inputs):
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return inputs, output_ids


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/process/")
async def process(instruction: str = Form(...), image_file: UploadFile = File(...)):
    async with request_semaphore:
        return await _process(instruction, image_file)


async def _process(instruction: str, image_file: UploadFile):
    start_time = time.time()
    try:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        width, height = image.width, image.height

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=resize_factor,
            min_pixels=resize_min_pixels,
            max_pixels=resize_max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": instruction},
            ],
        }

        image_inputs, video_inputs = process_vision_info([system_message, user_message])
        text = processor.apply_chat_template([system_message, user_message], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        async with inference_lock:
            inputs, output_ids = await asyncio.to_thread(run_inference, inputs)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        del inputs, output_ids, generated_ids, image_inputs, video_inputs

        logger.info("model raw output: %s", output_text)

        # Extract normalized coordinates (0~1) and convert to actual pixels
        norm_x, norm_y = extract_coordinates(output_text)
        pred_x = norm_x * width
        pred_y = norm_y * height

        elapsed = time.time() - start_time
        logger.info("request processed in %.3fs | image=%dx%d | instruction=%s | result=(%s, %s)", elapsed, width, height, instruction, pred_x, pred_y)
        return {"x": pred_x, "y": pred_y}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("inference failed in %.3fs: %s\n%s", elapsed, e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
