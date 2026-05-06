import argparse
import os
import time

import requests
from PIL import Image, ImageDraw, ImageFont


def test_health(base_url):
    resp = requests.get(f"{base_url}/")
    print(f"[Health Check] status={resp.status_code} body={resp.json()}")
    assert resp.status_code == 200


def test_process(base_url, image_path, instruction):
    with open(image_path, "rb") as f:
        start = time.time()
        resp = requests.post(
            f"{base_url}/process/",
            data={"instruction": instruction},
            files={"image_file": (image_path, f, "image/png")},
        )
        elapsed = time.time() - start
    result = resp.json()
    print(f"[Process] status={resp.status_code} result={result} time={elapsed:.3f}s")
    assert resp.status_code == 200
    assert "x" in result and "y" in result
    return result["x"], result["y"], elapsed


def annotate(image_path, x, y, instruction, elapsed, output_dir=None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    r = 18  # outer ring radius
    cross = 26  # crosshair half-length

    # Outer ring (red)
    draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
    # Crosshair
    draw.line((x - cross, y, x + cross, y), fill="red", width=2)
    draw.line((x, y - cross, x, y + cross), fill="red", width=2)
    # Center dot (yellow on red for contrast)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow", outline="red")

    # Label near the marker
    label = f"({x:.1f}, {y:.1f})  {elapsed:.2f}s\n{instruction}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Place label below-right of marker, but flip to above-left if it would go off-screen
    tx, ty = x + r + 8, y + r + 8
    bbox = draw.textbbox((tx, ty), label, font=font)
    if bbox[2] > img.width:
        tx = x - r - 8 - (bbox[2] - bbox[0])
    if bbox[3] > img.height:
        ty = y - r - 8 - (bbox[3] - bbox[1])
    bbox = draw.textbbox((tx, ty), label, font=font)
    # Background for readability
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill="black")
    draw.text((tx, ty), label, fill="white", font=font)

    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    safe_inst = "".join(c if c.isalnum() else "_" for c in instruction)[:40]
    out_name = f"{name}__{safe_inst}_marked{ext}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
    else:
        out_path = os.path.join(os.path.dirname(image_path), out_name)
    img.save(out_path)
    print(f"[Annotated] saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="test_files/4.png")
    parser.add_argument("instruction", nargs="?", default="寻梦之旅")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    print(f"Testing with url={args.url}, image={args.image}, instruction='{args.instruction}'")
    test_health(args.url)
    x, y, elapsed = test_process(args.url, args.image, args.instruction)
    annotate(args.image, x, y, args.instruction, elapsed, args.output_dir)
    print("All tests passed.")
