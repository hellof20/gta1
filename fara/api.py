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

# --- Tool Description Template ---
TOOL_DESC_TEMPLATE = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* The screen's resolution is {width}x{height}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant.

You are a web automation agent that performs actions on websites to fulfill user requests by calling various tools.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "computer_use", "description": "{tool_desc}", "parameters": {{"properties": {{"action": {{"description": "The action to perform.", "enum": ["left_click"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}}}}, "required": ["action"], "type": "object"}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_coordinates(raw_string):
    try:
        match = TOOL_CALL_PATTERN.search(raw_string)
        if not match:
            return 0, 0
        action_text = match.group(1).strip()
        action_json = json.loads(action_text)
        args = action_json.get("arguments", {})
        coordinate = args.get("coordinate", [])
        if len(coordinate) == 2:
            return coordinate[0], coordinate[1]
        return 0, 0
    except Exception:
        return 0, 0


# --- FastAPI App Initialization ---
app = FastAPI()

# --- Model and Processor Loading (on startup) ---
model_path = "microsoft/Fara-7B"
max_new_tokens = 256

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
    max_pixels=12845056,
    local_files_only=True,
)

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
        scale_x = width / resized_width
        scale_y = height / resized_height

        # Build system prompt with dynamic screen resolution
        tool_desc = TOOL_DESC_TEMPLATE.format(width=resized_width, height=resized_height)
        tool_desc_escaped = tool_desc.replace('"', '\\"').replace('\n', '\\n')
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_desc=tool_desc_escaped)

        system_message = {
            "role": "system",
            "content": system_prompt,
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": f"Click on: {instruction}"},
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

        # Extract coordinates (pixel values relative to resized image) and scale back
        pred_x, pred_y = extract_coordinates(output_text)
        pred_x *= scale_x
        pred_y *= scale_y

        elapsed = time.time() - start_time
        logger.info("request processed in %.3fs | image=%dx%d | instruction=%s | result=(%s, %s)", elapsed, width, height, instruction, pred_x, pred_y)
        return {"x": pred_x, "y": pred_y}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("inference failed in %.3fs: %s\n%s", elapsed, e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
