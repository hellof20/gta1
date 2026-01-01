from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import re
import io
import uvicorn

# --- System Prompt and Coordinate Extraction ---
SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()

def extract_coordinates(raw_string):
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return 0, 0

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Model and Processor Loading (on startup) ---
model_path = "HelloKKMe/GTA1-7B"
max_new_tokens = 32

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=3136,
    max_pixels=4096 * 2160
)

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/process/")
async def process(instruction: str = Form(...), image_file: UploadFile = File(...)):
    # Read and process the uploaded image
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    width, height = image.width, image.height

    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height))
    scale_x, scale_y = width / resized_width, height / resized_height

    # Prepare messages for the model
    system_message = {
       "role": "system",
       "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": resized_image},
            {"type": "text", "text": instruction}
        ]
    }

    # Tokenize and prepare inputs
    image_inputs, video_inputs = process_vision_info([system_message, user_message])
    text = processor.apply_chat_template([system_message, user_message], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate prediction
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, use_cache=True)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # Extract and rescale coordinates
    pred_x, pred_y = extract_coordinates(output_text)
    pred_x *= scale_x
    pred_y *= scale_y

    return {"x": pred_x, "y": pred_y}

# --- Main block to run the app ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)