#!/usr/bin/env python3
"""Analyze gripper openness statistics across the dataset.

Generates per-object, per-phase, and per-suite breakdowns of gripper openness
values to understand the relationship between object size and finger gap.

Usage:
  python scripts/stats_openness.py --data_dir data/gt-demo-libero-all-suite-train-0515
  python scripts/stats_openness.py --data_dir data/gt-demo-libero-all-suite-train-0515 --out stat/stats_openness.md
"""
from __future__ import annotations

import argparse
import json
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory containing GT JSON files (with subdirectories)")
    p.add_argument("--out", default=None,
                   help="Output markdown file (default: print to stdout)")
    return p.parse_args()


def load_samples(data_dir: str) -> list[dict]:
    """Load all GT JSON samples from a directory tree."""
    files = sorted(glob.glob(f"{data_dir}/**/sample_*_traj_*.json", recursive=True))
    samples = []
    for f in files:
        d = json.load(open(f))
        gt = d.get("gt", {})
        samples.append({
            "suite": d.get("suite", ""),
            "task_id": d.get("task_id", -1),
            "task_description": d.get("task_description", ""),
            "traj_step": d.get("traj_step", 0),
            "traj_len": d.get("traj_len", 0),
            "target_object": gt.get("target_object_name", ""),
            "openness": gt.get("gripper_openness"),
            "phase": gt.get("gripper_phase", ""),
            "can_close": gt.get("can_close", False),
            "finger_contact": gt.get("finger_contact", False),
            "action_gripper": gt.get("demo_action", [0] * 7)[6],
        })
    return samples


def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def stats_dict(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p25": None, "median": None, "p75": None}
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
    }


def main():
    args = parse_args()
    samples = load_samples(args.data_dir)
    print(f"Loaded {len(samples)} samples from {args.data_dir}")

    lines = []
    def pr(s=""):
        lines.append(s)
        print(s)

    pr(f"# Gripper Openness Statistics")
    pr(f"")
    pr(f"Data: `{args.data_dir}`  |  Total samples: {len(samples)}")
    pr()

    # ── By gripper phase ──────────────────────────────────────────────
    pr("## By Gripper Phase")
    pr()
    pr(f"| {'Phase':12s} | {'N':>5s} | {'Mean':>6s} | {'Std':>6s} | {'Min':>6s} | {'P25':>6s} | {'Med':>6s} | {'P75':>6s} | {'Max':>6s} |")
    pr(f"|{'-'*14}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|")
    by_phase = defaultdict(list)
    for s in samples:
        if s["openness"] is not None:
            by_phase[s["phase"]].append(s["openness"])
    for phase in ["approaching", "grasping", "carrying", "releasing", ""]:
        if phase in by_phase:
            st = stats_dict(by_phase[phase])
            label = phase if phase else "(unknown)"
            pr(f"| {label:12s} | {st['n']:>5d} | {fmt(st['mean']):>6s} | {fmt(st['std']):>6s} | {fmt(st['min']):>6s} | {fmt(st['p25']):>6s} | {fmt(st['median']):>6s} | {fmt(st['p75']):>6s} | {fmt(st['max']):>6s} |")
    pr()

    # ── By target object (during close command only) ──────────────────
    pr("## By Target Object (during close command, action[6] > 0)")
    pr()
    pr(f"| {'Object':45s} | {'N':>5s} | {'Mean':>6s} | {'Std':>6s} | {'Min':>6s} | {'Med':>6s} | {'Max':>6s} |")
    pr(f"|{'-'*47}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|")

    by_object = defaultdict(list)
    for s in samples:
        if s["openness"] is not None and s["action_gripper"] > 0:
            by_object[s["target_object"]].append(s["openness"])

    # Sort by median openness (large objects first)
    for obj in sorted(by_object.keys(), key=lambda k: -np.median(by_object[k])):
        st = stats_dict(by_object[obj])
        pr(f"| {obj:45s} | {st['n']:>5d} | {fmt(st['mean']):>6s} | {fmt(st['std']):>6s} | {fmt(st['min']):>6s} | {fmt(st['median']):>6s} | {fmt(st['max']):>6s} |")
    pr()

    # ── By target object (carrying phase only — stable grasp) ─────────
    pr("## By Target Object (carrying phase only — stable grasp)")
    pr()
    pr(f"| {'Object':45s} | {'N':>5s} | {'Mean':>6s} | {'Std':>6s} | {'Min':>6s} | {'Med':>6s} | {'Max':>6s} |")
    pr(f"|{'-'*47}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|")

    by_object_carry = defaultdict(list)
    for s in samples:
        if s["openness"] is not None and s["phase"] == "carrying":
            by_object_carry[s["target_object"]].append(s["openness"])

    for obj in sorted(by_object_carry.keys(), key=lambda k: -np.median(by_object_carry[k])):
        st = stats_dict(by_object_carry[obj])
        pr(f"| {obj:45s} | {st['n']:>5d} | {fmt(st['mean']):>6s} | {fmt(st['std']):>6s} | {fmt(st['min']):>6s} | {fmt(st['median']):>6s} | {fmt(st['max']):>6s} |")
    pr()

    # ── By suite ──────────────────────────────────────────────────────
    pr("## By Suite")
    pr()
    pr(f"| {'Suite':20s} | {'N':>5s} | {'Approach':>8s} | {'Grasp':>8s} | {'Carry':>8s} | {'Release':>8s} |")
    pr(f"|{'-'*22}|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")

    by_suite_phase = defaultdict(lambda: defaultdict(list))
    for s in samples:
        if s["openness"] is not None:
            by_suite_phase[s["suite"]][s["phase"]].append(s["openness"])

    for suite in sorted(by_suite_phase.keys()):
        phases = by_suite_phase[suite]
        n = sum(len(v) for v in phases.values())
        ap = fmt(np.mean(phases["approaching"])) if phases["approaching"] else "N/A"
        gr = fmt(np.mean(phases["grasping"])) if phases["grasping"] else "N/A"
        ca = fmt(np.mean(phases["carrying"])) if phases["carrying"] else "N/A"
        re = fmt(np.mean(phases["releasing"])) if phases["releasing"] else "N/A"
        pr(f"| {suite:20s} | {n:>5d} | {ap:>8s} | {gr:>8s} | {ca:>8s} | {re:>8s} |")
    pr()

    # ── Save ──────────────────────────────────────────────────────────
    if args.out:
        Path(args.out).write_text("\n".join(lines))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
