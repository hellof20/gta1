import time
SCRIPT_START = time.perf_counter()

import io
import logging
import os
import re
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import REGISTRY, Counter, Gauge, make_asgi_app
from qwen_vl_utils import smart_resize
from transformers import AutoImageProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# --- transformers compat shim ---------------------------------------------
# OpenCUA's tokenization_opencua.py does
#   `from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode`
# which newer transformers (5.x) no longer exposes there. Re-inject the
# (stable, well-known) GPT-2 byte->unicode map so the remote tokenizer code
# can import it. Must run before AutoTokenizer.from_pretrained().
import transformers.models.gpt2.tokenization_gpt2 as _gpt2_tok  # noqa: E402
if not hasattr(_gpt2_tok, "bytes_to_unicode"):
    def _bytes_to_unicode():
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))
    _gpt2_tok.bytes_to_unicode = _bytes_to_unicode
# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("vllm").setLevel(logging.WARNING)
logger.info("STARTUP imports loaded in %.2fs", time.perf_counter() - SCRIPT_START)

# --- relax transformers >=4.56 processor tokenizer-class check -------------
# This GTA1 checkpoint is registered as qwen2_5_vl, so vLLM builds a
# Qwen2_5_VLProcessor, but ships OpenCUA's TikTokenV3 tokenizer (correct for
# the model). transformers 4.56 strictly rejects that pairing, which only
# blocks vLLM from reading the Qwen2.5-VL *image* processor it needs. Relax
# the check so the processor builds; the tiktoken vocab already contains the
# vision/image tokens, so multimodal processing stays consistent.
try:
    import transformers.processing_utils as _proc_utils
    _ProcMixin = _proc_utils.ProcessorMixin
    if hasattr(_ProcMixin, "check_argument_for_proper_class"):
        _orig_proper_class_check = _ProcMixin.check_argument_for_proper_class

        def _lenient_proper_class_check(self, argument_name, arg, *a, **k):
            try:
                return _orig_proper_class_check(self, argument_name, arg, *a, **k)
            except (TypeError, ValueError) as e:
                logger.warning("relaxing processor class check for %s: %s", argument_name, e)

        _ProcMixin.check_argument_for_proper_class = _lenient_proper_class_check
except Exception as e:  # noqa: BLE001
    logger.warning("could not patch processor class check: %s", e)
# --------------------------------------------------------------------------

# --- make OpenCUA processor annotations resolvable ------------------------
# processing_opencua.py annotates processor kwargs with PILImageResampling but
# doesn't import it at runtime (it's under TYPE_CHECKING), so vLLM's
# get_type_hints() introspection eval()s that annotation and raises a NameError
# ("Failed to collect processor kwargs"). It's non-fatal but noisy. Exposing the
# symbol as a builtin lets get_type_hints resolve it in any module, regardless of
# import timing — this removes the error rather than just hiding the log.
try:
    import builtins as _builtins
    from transformers.image_utils import PILImageResampling as _PILResample
    _builtins.PILImageResampling = _PILResample
except Exception as e:  # noqa: BLE001
    logger.warning("could not expose PILImageResampling for processor introspection: %s", e)
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# This targets the NEW Salesforce/GTA1-7B (OpenCUA-based, "GUI agent" format).
# It is NOT compatible with the original HelloKKMe/GTA1-7B grounding model
# served by ../gta-1/api_vllm.py. Key differences vs gta-1:
#   - agent system prompt (no resolution injected) instead of the locator prompt
#   - model emits `pyautogui.click(x=, y=)` instead of a bare `(x,y)` pair
#   - OpenCUA tiktoken tokenizer -> needs tokenizer_mode="slow" + trust_remote_code
#   - chat template emits <|media_begin|>..<|media_end|>, which we rewrite to the
#     standard Qwen vision tokens that vLLM's qwen2_5_vl path expects
#   - max_new_tokens 512 (full action text) instead of 32
# If the original grounding model is mounted here by mistake, the template will
# not emit media tokens and build_prompt() raises -> fail fast at warmup.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)

# OpenCUA chat template wraps the image in a media block; rewrite to the vision
# tokens the underlying Qwen2.5-VL graph (used by vLLM) actually consumes.
MEDIA_BLOCK = re.compile(r"<\|media_begin\|>.*?<\|media_end\|>", re.S)
VISION_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"

# Model outputs full pyautogui code; pull the first click out of it.
CLICK_REGEXES = [
    re.compile(r"click\s*\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\)", re.IGNORECASE),
    re.compile(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
]
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


def extract_click(raw: str):
    """Return (x, y) in resized-image space, or None if no click action."""
    if "click" not in raw.lower():
        return None
    for rx in CLICK_REGEXES:
        if m := rx.search(raw):
            return int(m.group(1)), int(m.group(2))
    logger.warning("no click coords found in model output: %s", raw)
    return None


# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "Salesforce/GTA1-7B")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "12288"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
# Hard cap on concurrent /process requests; above this we 429.
INFLIGHT_LIMIT = int(os.environ.get("INFLIGHT_LIMIT", "4"))
ENABLE_PREFIX_CACHE = os.environ.get("ENABLE_PREFIX_CACHE", "1").lower() in ("1", "true", "yes", "on")
ENABLE_MM_CACHE = os.environ.get("ENABLE_MM_CACHE", "1").lower() in ("1", "true", "yes", "on")
# min/max pixels: default to the values shipped with the model's image processor,
# overridable via env (parsed after the processor loads in lifespan).
# The model's native max_pixels (~12.8M px ≈ 16k image tokens) makes vLLM reserve
# far too much memory during profiling, so we cap it. 3840*2160 keeps parity with
# ../gta-1 and fits a 24GB GPU. Override via MAX_PIXELS.
_MIN_PIXELS_ENV = os.environ.get("MIN_PIXELS")
_MAX_PIXELS_ENV = os.environ.get("MAX_PIXELS")
_MAX_PIXELS_DEFAULT = 3840 * 2160

# --- instruction normalization --------------------------------------------
# Upstream (qirabot decision planner) sends a bare element DESCRIPTION as the
# instruction, e.g. "开始游戏按钮" — in that system the action type
# (click/double-click/hover/type/drag) is decided separately and applied by the
# caller using the (x,y) we return, so the grounding model is a pure coordinate
# resolver. The OpenCUA/agent model instead expects an actionable task and emits
# a pyautogui action. So we always wrap the description as a *click* task — never
# mirror the caller's real action — because a click is the most reliable way to
# elicit a clean `pyautogui.click(x,y)` we can parse, and the caller applies the
# real action to the returned point. Set CLICK_PREFIX="" to disable wrapping.
CLICK_PREFIX = os.environ.get("CLICK_PREFIX", "点击")
# If the instruction already starts with one of these verbs, don't prefix again.
_CLICK_VERBS = ("点击", "单击", "点选", "click", "tap")


def normalize_instruction(instruction: str) -> str:
    instr = instruction.strip()
    if not CLICK_PREFIX or not instr:
        return instr
    if instr.lower().startswith(_CLICK_VERBS):
        return instr
    return f"{CLICK_PREFIX}{instr}"

# --- Runtime globals (initialized in lifespan) ---
engine = None
tokenizer = None
RESIZE_FACTOR = None
MIN_PIXELS = None
MAX_PIXELS = None
SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

_ready = False


def build_prompt(instruction: str) -> str:
    """Render the chat template and rewrite OpenCUA media tokens to vision tokens.

    Raises if the template does not emit a media block — that means the mounted
    model is not the OpenCUA/agent variant this server expects.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": "placeholder"},
            {"type": "text", "text": instruction},
        ]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text, n = MEDIA_BLOCK.subn(VISION_TOKENS, text)
    if n == 0:
        raise RuntimeError(
            "chat template did not emit <|media_begin|>..<|media_end|>; the mounted "
            f"model at {MODEL_PATH!r} is not the OpenCUA/agent GTA1 variant this "
            "server expects (the grounding model belongs in ../gta-1 instead)"
        )
    return text


@asynccontextmanager
async def lifespan(app):
    global engine, tokenizer, RESIZE_FACTOR, MIN_PIXELS, MAX_PIXELS, _ready

    _t = time.perf_counter()
    # OpenCUA ships a custom tokenizer/processor -> slow tokenizer + remote code.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    img_proc = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    RESIZE_FACTOR = img_proc.patch_size * img_proc.merge_size
    MIN_PIXELS = int(_MIN_PIXELS_ENV) if _MIN_PIXELS_ENV else img_proc.min_pixels
    MAX_PIXELS = int(_MAX_PIXELS_ENV) if _MAX_PIXELS_ENV else min(img_proc.max_pixels, _MAX_PIXELS_DEFAULT)
    logger.info("STARTUP tokenizer/processor ready in %.2fs (factor=%d min_px=%d max_px=%d)",
                time.perf_counter() - _t, RESIZE_FACTOR, MIN_PIXELS, MAX_PIXELS)

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        tokenizer_mode="slow",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        # We pre-resize the image ourselves; keep vLLM's bounds in sync so its
        # internal qwen2_5_vl processor is a no-op rather than resizing again.
        mm_processor_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=ENABLE_PREFIX_CACHE,
        mm_processor_cache_gb=(4 if ENABLE_MM_CACHE else 0),
        enforce_eager=False,
    )
    _t = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("STARTUP engine ready in %.2fs", time.perf_counter() - _t)

    _t = time.perf_counter()
    warmup_prompt = build_prompt("button")  # also asserts the model is the agent variant
    warmup_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    async for _ in engine.generate(
        prompt={"prompt": warmup_prompt, "multi_modal_data": {"image": warmup_img}},
        sampling_params=SAMPLING,
        request_id="warmup",
    ):
        pass
    logger.info("STARTUP warmup done in %.2fs", time.perf_counter() - _t)

    elapsed = time.perf_counter() - SCRIPT_START
    bar = "=" * 60
    logger.info(bar)
    logger.info("STARTUP COMPLETE — server ready after %.2fs from script start", elapsed)
    logger.info(bar)
    _ready = True
    yield


app = FastAPI(lifespan=lifespan)

INFLIGHT = Gauge("gta1_inflight_requests", "Currently in-flight /process requests")
REJECTED = Counter("gta1_rejected_requests_total", "Requests rejected due to inflight limit")
_inflight = 0  # asyncio is single-threaded, no lock needed


@app.middleware("http")
async def limit_inflight(request: Request, call_next):
    global _inflight
    if request.url.path != "/process/":
        return await call_next(request)
    if _inflight >= INFLIGHT_LIMIT:
        REJECTED.inc()
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "2"},
            content={"error": "too many requests, retry later"},
        )
    _inflight += 1
    INFLIGHT.set(_inflight)
    try:
        return await call_next(request)
    finally:
        _inflight -= 1
        INFLIGHT.set(_inflight)


app.mount("/metrics", make_asgi_app(REGISTRY))


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.get("/ready")
async def ready_check():
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ready"}


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

        # Pre-resize to the dims the model reasons in, then scale coords back.
        t = time.perf_counter()
        resized_height, resized_width = smart_resize(
            height, width, factor=RESIZE_FACTOR,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        resized_image = image.resize((resized_width, resized_height))
        scale_x, scale_y = width / resized_width, height / resized_height
        resize_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        task = normalize_instruction(instruction)
        prompt_text = build_prompt(task)
        preprocess_ms = (time.perf_counter() - t) * 1000

        gen_t = time.perf_counter()
        request_id = uuid.uuid4().hex
        results = engine.generate(
            prompt={"prompt": prompt_text, "multi_modal_data": {"image": resized_image}},
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

        n_tok = len(final_output.outputs[0].token_ids) if final_output and final_output.outputs else 0
        if ttft_ms and n_tok > 1:
            decode_avg_ms = (gen_ms - ttft_ms) / (n_tok - 1)
            prefill_ms = max(ttft_ms - decode_avg_ms, 0.0)
        else:
            decode_avg_ms = 0.0
            prefill_ms = ttft_ms or 0.0

        t = time.perf_counter()
        output_text = final_output.outputs[0].text
        gen_tokens = len(final_output.outputs[0].token_ids)
        click = extract_click(output_text)
        if click is None:
            pred_x = pred_y = None
        else:
            pred_x, pred_y = click[0] * scale_x, click[1] * scale_y
        post_ms = (time.perf_counter() - t) * 1000

        total_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "TIMING total=%.1fms | read=%.1f decode=%.1f resize=%.1f preprocess=%.1f "
            "generate=%.1f(prefill~%.1f decode_avg=%.1f) post=%.1f "
            "| gen_tok=%d | %dx%d->%dx%d | %r -> %r -> %s",
            total_ms, read_ms, decode_ms, resize_ms, preprocess_ms,
            gen_ms, prefill_ms, decode_avg_ms, post_ms,
            gen_tokens, width, height, resized_width, resized_height, instruction, task,
            (pred_x, pred_y),
        )
        return {"x": pred_x, "y": pred_y, "raw": output_text}

    except Exception as e:
        logger.error("inference failed in %.3fs: %s\n%s",
                     time.perf_counter() - start_time, e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": "internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
