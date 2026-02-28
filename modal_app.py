import modal

MODEL_ID = "HelloKKMe/GTA1-7B"
VOLUME_PATH = "/model-cache"

app = modal.App("gta1-inference")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "accelerate==1.1.1",
        "Pillow==11.1.0",
        "transformers==4.51.3",
        "qwen-vl-utils==0.0.8",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
    )
    .pip_install(
        "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .env({
        "HF_HOME": VOLUME_PATH,
        "TORCHINDUCTOR_CACHE_DIR": f"{VOLUME_PATH}/torch_cache",
    })
)

model_volume = modal.Volume.from_name("gta1-model-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={VOLUME_PATH: model_volume},
    timeout=1800,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    model_volume.commit()


def create_app(inference):
    import re
    import io
    import time
    import asyncio
    import logging
    from fastapi import FastAPI, File, Form, UploadFile
    from PIL import Image
    from qwen_vl_utils import process_vision_info, smart_resize
    import torch

    logger = logging.getLogger("gta1")

    SYSTEM_PROMPT = (
        "You are an expert UI element locator. Given a GUI image and a user's "
        "element description, provide the coordinates of the specified element "
        "as a single (x,y) point. The image resolution is height {height} and "
        "width {width}. For elements with area, return the center point.\n\n"
        "Output the coordinate pair exactly:\n(x,y)"
    )

    COORD_PATTERN = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")

    def extract_coordinates(raw_string):
        try:
            matches = COORD_PATTERN.findall(raw_string)
            return [tuple(map(int, match)) for match in matches][0]
        except Exception:
            return 0, 0

    web_app = FastAPI()
    inference_lock = asyncio.Lock()

    def run_inference(inputs):
        inputs = inputs.to(inference.model.device)
        with torch.inference_mode():
            output_ids = inference.model.generate(
                **inputs,
                max_new_tokens=inference.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return inputs, output_ids

    @web_app.get("/")
    async def health_check():
        return {"status": "healthy"}

    @web_app.post("/process/")
    async def process(
        instruction: str = Form(...),
        image_file: UploadFile = File(...),
    ):
        start_time = time.time()
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        width, height = image.width, image.height

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=inference.resize_factor,
            min_pixels=inference.resize_min_pixels,
            max_pixels=inference.resize_max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))
        scale_x = width / resized_width
        scale_y = height / resized_height

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width),
        }
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": instruction},
            ],
        }

        image_inputs, video_inputs = process_vision_info([system_message, user_message])
        text = inference.processor.apply_chat_template(
            [system_message, user_message],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = inference.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        async with inference_lock:
            inputs, output_ids = await asyncio.to_thread(run_inference, inputs)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = inference.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        del inputs, output_ids, generated_ids, image_inputs, video_inputs

        pred_x, pred_y = extract_coordinates(output_text)
        pred_x *= scale_x
        pred_y *= scale_y

        elapsed = time.time() - start_time
        logger.info(
            "request processed in %.3fs | image=%dx%d | instruction=%s | result=(%s, %s)",
            elapsed, width, height, instruction, pred_x, pred_y,
        )
        return {"x": pred_x, "y": pred_y}

    return web_app


@app.cls(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: model_volume},
    scaledown_window=300,
    timeout=120,
)
@modal.concurrent(max_inputs=10)
class Inference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import logging

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("gta1")
        self.logger.info("loading model %s", MODEL_ID)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=3136,
            max_pixels=4096 * 2160,
        )
        self.model = torch.compile(self.model)

        self.resize_factor = (
            self.processor.image_processor.patch_size
            * self.processor.image_processor.merge_size
        )
        self.resize_min_pixels = self.processor.image_processor.min_pixels
        self.resize_max_pixels = self.processor.image_processor.max_pixels
        self.max_new_tokens = 32

        model_volume.commit()
        self.logger.info("model loaded successfully")

    @modal.asgi_app()
    def serve(self):
        return create_app(self)


@app.local_entrypoint()
def main():
    download_model.remote()
    print("Model downloaded successfully. Deploy with: modal deploy modal_app.py")
