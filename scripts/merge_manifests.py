#!/usr/bin/env python3
"""Merge per-suite manifest.json files into a top-level manifest.json.

Usage:
  python scripts/merge_manifests.py --out_dir data/gt-q11-libero-train
"""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, help="Top-level GT directory (e.g. data/gt-q11-libero-train)")
    args = p.parse_args()

    base = Path(args.out_dir)
    all_records = []

    for suite_manifest in sorted(base.glob("*/manifest.json")):
        records = json.loads(suite_manifest.read_text())
        print(f"  {suite_manifest.parent.name}: {len(records)} samples")
        all_records.extend(records)

    # Re-assign sequential sample_id
    for i, r in enumerate(all_records):
        r["sample_id"] = i

    top_manifest = base / "manifest.json"
    top_manifest.write_text(json.dumps(all_records, indent=2))
    print(f"\nTotal: {len(all_records)} samples -> {top_manifest}")

if __name__ == "__main__":
    main()
