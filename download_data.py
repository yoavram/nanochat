import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

FILES = [
    # "TinyStories-valid.txt",        # 19.4 MB
    "TinyStoriesV2-GPT4-valid.txt", # 22.5 MB
    # "TinyStories-train.txt",      # 1.92 GB
    "TinyStoriesV2-GPT4-train.txt", # 2.23 GB
]

def download(filename: str, dest_dir: Path = Path(".")) -> None:
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
    dest_dir = Path("data")
    dest_dir.mkdir(exist_ok=True)
    for f in FILES:
        download(f, dest_dir)