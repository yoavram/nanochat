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

# The instruction-tuning corpus used by `nanochat-sft.ipynb`. It lives in a *different*
# HuggingFace repo, whose name has no hyphen (`TinyStoriesInstruct`) even though the
# files inside it do. The hyphenated repo name 404s.
INSTRUCT_VALID_FILE = "TinyStories-Instruct-valid.txt"  # 26.9 MB
INSTRUCT_TRAIN_FILE = "TinyStories-Instruct-train.txt"  # 2.66 GB

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
INSTRUCT_URL = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main"

def download_tinystories(filename: str, dest_dir: Path = Path("."),
                         base_url: str = TINYSTORIES_URL) -> None:
    url = f"{base_url}/{filename}?download=true"
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
    download_tinystories(INSTRUCT_VALID_FILE, dest_dir, INSTRUCT_URL)
    if "--train" in sys.argv:
        download_tinystories(TRAIN_FILE, dest_dir)
        download_tinystories(INSTRUCT_TRAIN_FILE, dest_dir, INSTRUCT_URL)
    else:
        print(f"Skipping {TRAIN_FILE} (2.23 GB) and {INSTRUCT_TRAIN_FILE} (2.66 GB)"
              f" — pass --train to download them.")