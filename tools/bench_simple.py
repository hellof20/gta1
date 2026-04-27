import requests
import sys
import time
import threading
import statistics

BASE_URL = "https://gta1-279432852451.asia-southeast1.run.app"

IMAGE_PATH = "images/screen.jpg"
INSTRUCTION = "开局直接亮刀子"
DURATION = 60
RPS = 0.5


def send_request(seq, image_path, instruction, results):
    try:
        start = time.time()
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/process/",
                data={"instruction": instruction},
                files={"image_file": ("image.png", f, "image/png")},
            )
        elapsed = time.time() - start
        status = resp.status_code
        results.append({"seq": seq, "status": status, "elapsed": elapsed, "error": None})
        print(f"  #{seq:03d}  status={status}  elapsed={elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        results.append({"seq": seq, "status": 0, "elapsed": elapsed, "error": str(e)})
        print(f"  #{seq:03d}  ERROR: {e}")


def main():
    image_path = sys.argv[1] if len(sys.argv) >= 2 else IMAGE_PATH
    instruction = sys.argv[2] if len(sys.argv) >= 3 else INSTRUCTION
    duration = int(sys.argv[3]) if len(sys.argv) >= 4 else DURATION

    print(f"bench config:")
    print(f"  url:         {BASE_URL}")
    print(f"  image:       {image_path}")
    print(f"  instruction: {instruction}")
    print(f"  duration:    {duration}s")
    print(f"  rate:        {RPS} req/s")
    print()

    results = []
    threads = []

    print("sending requests...")
    total_requests = int(duration * RPS)
    for i in range(total_requests):
        t = threading.Thread(target=send_request, args=(i + 1, image_path, instruction, results))
        t.start()
        threads.append(t)
        if i < total_requests - 1:
            time.sleep(1.0 / RPS)

    print("\nwaiting for all requests to complete...")
    for t in threads:
        t.join()

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    total = len(results)
    success = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]
    elapsed_list = [r["elapsed"] for r in results]
    success_elapsed = [r["elapsed"] for r in success]

    print(f"  total requests:  {total}")
    print(f"  success (200):   {len(success)}")
    print(f"  failed:          {len(failed)}")

    if elapsed_list:
        print(f"\n  all requests:")
        print(f"    min:    {min(elapsed_list):.2f}s")
        print(f"    max:    {max(elapsed_list):.2f}s")
        print(f"    avg:    {statistics.mean(elapsed_list):.2f}s")
        print(f"    median: {statistics.median(elapsed_list):.2f}s")
        if len(elapsed_list) >= 2:
            print(f"    stdev:  {statistics.stdev(elapsed_list):.2f}s")

    if success_elapsed:
        print(f"\n  success requests:")
        print(f"    min:    {min(success_elapsed):.2f}s")
        print(f"    max:    {max(success_elapsed):.2f}s")
        print(f"    avg:    {statistics.mean(success_elapsed):.2f}s")
        print(f"    median: {statistics.median(success_elapsed):.2f}s")

    if failed:
        print(f"\n  failed details:")
        for r in failed:
            print(f"    #{r['seq']:03d}  status={r['status']}  error={r['error']}")


if __name__ == "__main__":
    main()
