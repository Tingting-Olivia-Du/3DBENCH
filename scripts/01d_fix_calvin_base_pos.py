#!/usr/bin/env python3
"""Backfill the correct robot base position in an existing CALVIN GT manifest.

Earlier `01b_extract_gt_calvin.py` runs hard-coded `gt['base_pos_world']` to
`[0.0, 0.0, 0.0]`. The real CALVIN env-D Panda base lives at
`[-0.34, -0.46, 0.24]` (per `validation/.hydra/merged_config.yaml`). This
field is metadata only — no metric script consumes it — but we keep the GT
honest by rewriting it in-place.

What's rewritten:
  - <gt_root>/manifest.json                 (top-level flat manifest)
  - <gt_root>/<suite>/manifest.json         (per-suite manifests)
  - <gt_root>/<suite>/task_*/window_*/sample_*.json   (per-sample JSONs)

Only the `gt['base_pos_world']` field is touched. Everything else is left
byte-for-byte identical (we do an in-place dict mutation and re-dump with the
same indent=2 formatting that 01b uses).

Usage:
  python scripts/01d_fix_calvin_base_pos.py
  python scripts/01d_fix_calvin_base_pos.py --gt_root data/gt-q6-mv-calvin --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from bench.calvin_taxonomy import ROBOT_BASE_POS_WORLD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_root", default="data/gt-q6-mv-calvin",
                   help="Root of the CALVIN GT dataset (contains manifest.json + <suite>/...).")
    p.add_argument("--dry_run", action="store_true",
                   help="Just count how many records would change; don't write anything.")
    return p.parse_args()


def _correct_value() -> list:
    return [float(x) for x in ROBOT_BASE_POS_WORLD.tolist()]


CORRECT = _correct_value()


def _needs_fix(rec: dict) -> bool:
    bp = rec.get("gt", {}).get("base_pos_world")
    if bp is None:
        return False
    if not isinstance(bp, list) or len(bp) != 3:
        return True
    return any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(bp, CORRECT))


def _fix_manifest_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Return (n_total, n_fixed) for one manifest.json."""
    if not path.exists():
        return 0, 0
    records = json.loads(path.read_text())
    n_total = len(records)
    n_fixed = 0
    for rec in records:
        if _needs_fix(rec):
            rec["gt"]["base_pos_world"] = list(CORRECT)
            n_fixed += 1
    if n_fixed > 0 and not dry_run:
        path.write_text(json.dumps(records, indent=2))
    return n_total, n_fixed


def _fix_sample_file(path: Path, dry_run: bool) -> bool:
    """Return True iff this sample JSON was patched."""
    rec = json.loads(path.read_text())
    if not _needs_fix(rec):
        return False
    rec["gt"]["base_pos_world"] = list(CORRECT)
    if not dry_run:
        path.write_text(json.dumps(rec, indent=2))
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.gt_root)
    if not root.exists():
        print(f"ERROR: gt_root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Correct base_pos_world: {CORRECT}")
    print(f"Scanning {root} ...  (dry_run={args.dry_run})")

    # 1) per-suite + top-level manifests
    manifest_paths = list(root.glob("*/manifest.json")) + ([root / "manifest.json"] if (root / "manifest.json").exists() else [])
    n_manifest_records = 0
    n_manifest_fixed = 0
    for mp in manifest_paths:
        total, fixed = _fix_manifest_file(mp, args.dry_run)
        print(f"  manifest {mp.relative_to(root)}: {fixed}/{total} records fixed")
        n_manifest_records += total
        n_manifest_fixed += fixed

    # 2) per-sample JSON files
    sample_paths = list(root.rglob("sample_*.json"))
    n_samples = len(sample_paths)
    n_samples_fixed = 0
    for i, sp in enumerate(sample_paths):
        if _fix_sample_file(sp, args.dry_run):
            n_samples_fixed += 1
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{n_samples} sample files scanned ({n_samples_fixed} fixed)")
    print(f"  sample JSONs: {n_samples_fixed}/{n_samples} fixed")

    print()
    print(f"Summary: manifests {n_manifest_fixed}/{n_manifest_records}, "
          f"sample files {n_samples_fixed}/{n_samples}")
    if args.dry_run:
        print("[dry_run] no files were written.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
