import asyncio
import io
import logging
import os
import re
import tempfile
import time
import traceback

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a GUI agent. Based on the UI screenshot provided, please locate the exact position "
    "of the element that matches the instruction given by the user.\n\n"
    "Requirements for the output:\n"
    "- Return only the point (x, y) representing the center of the target element\n"
    "- Coordinates must be normalized to the range [0, 1]\n"
    "- Round each coordinate to three decimal places\n"
    "- Format the output as strictly (x, y) without any additional text\n"
)

COORD_PATTERN = re.compile(r"\((\d+\.?\d*),\s*(\d+\.?\d*)\)")


def extract_coordinates(raw_string: str) -> tuple[float, float]:
    match = COORD_PATTERN.search(raw_string)
    if match:
        return float(match.group(1)), float(match.group(2))
    logger.warning("no coordinates found in model output: %s", raw_string)
    return 0.0, 0.0


app = FastAPI()

model_path = "tencent/POINTS-GUI-G"
max_new_tokens = 32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
    local_files_only=True,
)
logger.info("model loaded")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
image_processor = Qwen2VLImageProcessor.from_pretrained(model_path, local_files_only=True)

MAX_CONCURRENT_REQUESTS = 10
inference_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

GENERATION_CONFIG = {
    "max_new_tokens": max_new_tokens,
    "do_sample": False,
}


def run_inference(messages):
    return model.chat(messages, tokenizer, image_processor, GENERATION_CONFIG)


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/process/")
async def process(instruction: str = Form(...), image_file: UploadFile = File(...)):
    async with request_semaphore:
        return await _process(instruction, image_file)


async def _process(instruction: str, image_file: UploadFile):
    start_time = time.time()
    tmp_path = None
    try:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        width, height = image.width, image.height

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        async with inference_lock:
            output_text = await asyncio.to_thread(run_inference, messages)

        logger.info("model raw output: %s", output_text)

        norm_x, norm_y = extract_coordinates(output_text)
        pred_x = norm_x * width
        pred_y = norm_y * height

        elapsed = time.time() - start_time
        logger.info(
            "request processed in %.3fs | image=%dx%d | instruction=%s | result=(%s, %s)",
            elapsed, width, height, instruction, pred_x, pred_y,
        )
        return {"x": pred_x, "y": pred_y}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("inference failed in %.3fs: %s\n%s", elapsed, e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path:
            os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
