import requests
import sys
import time
import os
from PIL import Image, ImageDraw

# BASE_URL = "https://a15021127204--gta1-inference-inference-serve.modal.run" # MODAL
BASE_URL = "https://gta1-279432852451.asia-southeast1.run.app" # CLOUD_RUN L4
BASE_URL = "https://gta1-rtx6000-279432852451.us-central1.run.app" # CLOUD_RUN RTX6000


MARKER_RADIUS = 15
MARKER_COLOR = (255, 0, 0)
MARKER_WIDTH = 3


def test_health():
    print("testing health check...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"  status: {resp.status_code}")
    print(f"  response: {resp.json()}")
    assert resp.status_code == 200
    print("  passed\n")


def test_inference(image_path, instruction):
    print(f"testing inference...")
    print(f"  image: {image_path}")
    print(f"  instruction: {instruction}")

    start = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/process/",
            data={"instruction": instruction},
            files={"image_file": ("image.png", f, "image/png")},
        )
    elapsed = time.time() - start

    print(f"  status: {resp.status_code}")
    print(f"  elapsed: {elapsed:.2f}s")
    try:
        result = resp.json()
    except Exception:
        print(f"  raw response: {resp.text[:1000]}")
        raise
    print(f"  response: {result}")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}"

    x, y = int(result["x"]), int(result["y"])
    mark_image(image_path, x, y)
    print("  passed\n")


def mark_image(image_path, x, y):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    r = MARKER_RADIUS

    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        outline=MARKER_COLOR,
        width=MARKER_WIDTH,
    )
    draw.line([x - r, y, x + r, y], fill=MARKER_COLOR, width=MARKER_WIDTH)
    draw.line([x, y - r, x, y + r], fill=MARKER_COLOR, width=MARKER_WIDTH)

    name, ext = os.path.splitext(image_path)
    output_path = f"{name}_marked{ext}"
    img.save(output_path)
    print(f"  marked image saved to: {output_path}")


if __name__ == "__main__":
    test_health()

    if len(sys.argv) >= 3:
        test_inference(sys.argv[1], sys.argv[2])
    else:
        print("usage: python test.py <image_path> <instruction>")
        print('example: python test.py screenshot.png "find the submit button"')
