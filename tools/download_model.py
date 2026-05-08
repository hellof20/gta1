from huggingface_hub import snapshot_download
import os
import sys

MODEL_ID = "HelloKKMe/GTA1-7B"
# MODEL_ID = "tencent/POINTS-GUI-G"

def main():
    try:
        snapshot_download(MODEL_ID, local_files_only=True)
        print(f"Model '{MODEL_ID}' already exists locally, skipping download.")
    except FileNotFoundError:
        print(f"Model '{MODEL_ID}' not found locally, downloading...")
        snapshot_download(MODEL_ID)
        print(f"Model '{MODEL_ID}' downloaded successfully.")

if __name__ == "__main__":
    main()
