#!/usr/bin/env python3
"""Download and extract the PRCP-1010-InsClaimPred dataset."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATASET_URL = (
    "https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/"
    "PRCP-1010-InsClaimPred.zip"
)


def download_dataset(data_dir: Path, force: bool = False) -> Path:
    """Download the zip archive and extract train.csv into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    if train_csv.exists() and not force:
        print(f"Dataset already present: {train_csv}")
        return train_csv

    zip_path = data_dir / "PRCP-1010-InsClaimPred.zip"
    print(f"Downloading dataset from:\n  {DATASET_URL}")
    urlretrieve(DATASET_URL, zip_path)
    print(f"Saved archive to {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Archive contains Data/train.csv
        for member in zf.namelist():
            if member.endswith("train.csv"):
                with zf.open(member) as src, open(train_csv, "wb") as dst:
                    dst.write(src.read())
                break
        else:
            raise FileNotFoundError("train.csv not found inside the zip archive")

    print(f"Extracted training data to {train_csv}")
    return train_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory where train.csv will be saved",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if train.csv already exists",
    )
    args = parser.parse_args()
    download_dataset(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
