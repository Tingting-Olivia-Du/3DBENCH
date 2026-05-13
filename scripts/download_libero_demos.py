#!/usr/bin/env python3
"""Download LIBERO demo HDF5 files from HuggingFace.

Usage
-----
  # Download libero_10 demos (default)
  python scripts/download_libero_demos.py

  # Download a specific suite to a custom directory
  python scripts/download_libero_demos.py \
      --suite libero_spatial \
      --dest /path/to/datasets

  # Download multiple suites
  python scripts/download_libero_demos.py --suite libero_10,libero_spatial

  # Download all suites
  python scripts/download_libero_demos.py --suite all
"""
import argparse
from pathlib import Path

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
DEFAULT_DEST = "/workspace/tingting/LIBERO/libero/datasets"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--suite", default="all",
        help=(
            "Suite(s) to download, comma-separated or 'all'. "
            "Choices: libero_spatial, libero_object, libero_goal, "
            "libero_10, libero_90. (default: libero_10)"
        ),
    )
    p.add_argument(
        "--dest", default=DEFAULT_DEST,
        help=f"Destination directory for datasets (default: {DEFAULT_DEST})",
    )
    return p.parse_args()


def download_suite(suite_name: str, dest: str):
    from huggingface_hub import snapshot_download

    print(f"\n{'='*50}")
    print(f"Downloading {suite_name} → {dest}/{suite_name}/")
    print(f"{'='*50}")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=f"{suite_name}/*",
    )

    # Verify
    downloaded = list(Path(dest, suite_name).glob("*.hdf5"))
    print(f"  ✓ {len(downloaded)} HDF5 files downloaded")
    for f in sorted(downloaded):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    {f.name}  ({size_mb:.0f} MB)")


def main():
    args = parse_args()

    if args.suite.strip().lower() == "all":
        suites = ALL_SUITES
    else:
        suites = [s.strip() for s in args.suite.split(",") if s.strip()]

    unknown = [s for s in suites if s not in ALL_SUITES]
    if unknown:
        print(f"ERROR: Unknown suite(s): {unknown}")
        print(f"Available: {ALL_SUITES}")
        return

    dest = str(Path(args.dest).resolve())
    Path(dest).mkdir(parents=True, exist_ok=True)

    print(f"HuggingFace repo: {HF_REPO_ID}")
    print(f"Destination: {dest}")
    print(f"Suites: {suites}")

    for suite in suites:
        download_suite(suite, dest)

    print(f"\nDone! Files saved to {dest}")


if __name__ == "__main__":
    main()
