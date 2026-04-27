"""
Download the Capstone dataset (FaceForensics++ subset + still-image train/FAKE)
from the public Hugging Face repository

Usage
    python scripts/download_data.py 
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Toonlink/Capstone"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Toonlink/Capstone from Hugging Face")
    p.add_argument(
        "--dest",
        type=str,
        default=str(DEFAULT_DEST),
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=f"Hugging Face dataset repo id (default: {REPO_ID})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    print(f"Downloading {args.repo_id} -> {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )

    print("Done.")
    print("Top-level entries:")
    for entry in sorted(dest.iterdir()):
        print(f"  {entry.name}")


if __name__ == "__main__":
    main()
