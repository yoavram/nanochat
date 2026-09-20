import urllib.request
from pathlib import Path
import os

# POKER

def download_poker(data_dir):
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/poker'
    files = {
        'poker-hand-training-true.data': f'{base_url}/poker-hand-training-true.data',
        'poker-hand-testing.data': f'{base_url}/poker-hand-testing.data',
    }
    
    for fname, url in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f'Downloading {fname}...')
            urllib.request.urlretrieve(url, path)
            print('Done.')
            
# TinyStories
# The notebooks default to the validation split; the 2.23 GB training split is an
# explicit opt-in: `python download_data.py --train`.
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"  # 22.5 MB
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"  # 2.23 GB

def download_tinystories(filename: str, dest_dir: Path = Path(".")) -> None:
    BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
    url = f"{BASE_URL}/{filename}?download=true"
    dest = dest_dir / filename
    if dest.exists():
        print(f"Skipping {filename} (already exists)")
        return

    print(f"Downloading {filename} ...")
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  {pct:.1f}%  ({downloaded/1e6:.1f} / {total_size/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r  Done → {dest}")

if __name__ == "__main__":
    import sys

    dest_dir = Path("data")
    dest_dir.mkdir(exist_ok=True)
    download_poker(dest_dir)
    download_tinystories(VALID_FILE, dest_dir)
    if "--train" in sys.argv:
        download_tinystories(TRAIN_FILE, dest_dir)
    else:
        print(f"Skipping {TRAIN_FILE} (2.23 GB) — pass --train to download it.")