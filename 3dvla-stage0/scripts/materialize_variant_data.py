#!/usr/bin/env python3
"""Materialize a variant spec (sft/variants/<v>.json) into ONE shuffled jsonl
that starVLA's llava_json loader consumes directly. Offline materialization
(vs a trainer-side weighted sampler) keeps the mixture reproducible and
auditable: the emitted file IS what the model sees, so every data audit we
have can run on it unchanged.

Per slice:
  - sample take_convs without replacement; epochs>1 (anchor x1.89) = full
    copies + sampled remainder, all explicit
  - fine3d template_filter (axis variants): turn-level surgery on packed
    conversations — keep only QA pairs whose template matches; move the
    "<image>...\nPREAMBLE" prefix onto the first kept pair when pair 0 is cut;
    drop conversations with no matching pair. Shortfall vs the spec'd take is
    refilled with extra embodied conversations (spec note: token exposure
    matched; exact token rebalance happens in the calibration step).
  - relative image paths -> absolute (embodied/generic3d/general roots)
  - emit key `image` (starVLA _build_messages contract), keep K/Ks/source/
    templates/slice for the focal-unify hook and in-training audits

Output: sft/materialized/<variant>.jsonl (+ .manifest.json sidecar)
Self-check at the end re-validates placeholder==image count and path sample.
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pack_sharegpt import PREAMBLE

SFT = Path("/workspace/tingting/3dvla-data/sft")
DATA = Path("/workspace/tingting/3dvla-data")

ROOTS = {  # file-prefix -> image root for relative paths
    "vlaser_vsi": DATA / "vlaser/spatial_data",
    "vlaser_vlm_3r": DATA / "vlaser/spatial_data",
    "vlaser_": DATA / "vlaser/grounding_data",
    "general_": DATA / "general",
}


def root_of(fname):
    for pre, root in ROOTS.items():
        if fname.startswith(pre):
            return root
    return None  # absolute-path slices (m1_/libero_)


def filter_conv(d, allowed):
    """Turn-level template surgery. Returns new record dict or None."""
    conv, tmpl = d["conversations"], d["templates"]
    keep = [i for i, t in enumerate(tmpl) if t in allowed]
    if not keep:
        return None
    if len(keep) == len(tmpl):
        return d
    first_q = conv[0]["value"]
    cut = first_q.find(PREAMBLE)
    assert cut >= 0, "packed conversation lost its preamble"
    prefix = first_q[: cut + len(PREAMBLE)]
    new_conv = []
    for j, i in enumerate(keep):
        q = conv[2 * i]["value"]
        if i == 0:
            body = q[cut + len(PREAMBLE):]
        else:
            body = q
        new_conv.append({"from": "human", "value": (prefix + body) if j == 0 else body})
        new_conv.append({"from": "gpt", "value": conv[2 * i + 1]["value"]})
    out = dict(d)
    out["conversations"] = new_conv
    out["templates"] = [tmpl[i] for i in keep]
    return out


def sample_indices(n_avail, take, rng):
    """Explicit-repetition sampling: full copies + sampled remainder."""
    idx = []
    while take >= n_avail:
        idx.extend(range(n_avail))
        take -= n_avail
    if take:
        idx.extend(rng.sample(range(n_avail), take))
    return idx


def load_slice(name, spec, rng, fine3d_filter):
    files = spec["files"]
    take = spec["take_convs"]
    if take == 0 or not files:
        return [], 0
    # pass 1: line offsets (+ survivor mask for filtered fine3d)
    entries = []  # (file, offset)
    for f in files:
        with open(SFT / f) as fh:
            off = fh.tell()
            line = fh.readline()
            while line:
                if fine3d_filter is not None:
                    d = json.loads(line)
                    if not any(t in fine3d_filter for t in d["templates"]):
                        off = fh.tell(); line = fh.readline(); continue
                entries.append((f, off))
                off = fh.tell(); line = fh.readline()
    n_avail = len(entries)
    actual = min(take, n_avail if fine3d_filter is not None else take)
    if fine3d_filter is not None and actual < take:
        print(f"  [{name}] filter {sorted(fine3d_filter)}: survivors {n_avail:,} < take {take:,}"
              f" -> shortfall {take - actual:,} refilled from embodied")
    picked = sample_indices(n_avail, actual, rng)
    # pass 2: group picked by (file, offset) to read each line once
    from collections import Counter
    cnt = Counter(picked)
    root_cache = {}
    out = []
    for i, (f, off) in enumerate(entries):
        c = cnt.get(i)
        if not c:
            continue
        with open(SFT / f) as fh:
            fh.seek(off)
            d = json.loads(fh.readline())
        if fine3d_filter is not None:
            d = filter_conv(d, fine3d_filter)
            if d is None:
                continue
        root = root_cache.setdefault(f, root_of(f))
        imgs = d.pop("images")
        if root is not None:
            imgs = [str(root / p) for p in imgs]
        rec = {"image": imgs, "conversations": d["conversations"],
               "K": d.get("K"), "Ks": d.get("Ks"),
               "source": d.get("source"), "templates": d.get("templates"),
               "slice": name}
        out.extend([rec] * c)
        # NB: repeated convs share the dict; json.dumps below copies content.
    return out, take - actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total convs (smoke runs); scales slices proportionally")
    ap.add_argument("--spec-dir", default="variants",
                    help="variants (conv-weighted) or variants_tokcal (token-calibrated)")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()
    spec = json.load(open(SFT / f"{args.spec_dir}/{args.variant}.json"))
    rng = random.Random(args.seed)
    out_dir = SFT / "materialized"
    out_dir.mkdir(exist_ok=True)

    scale = 1.0
    if args.limit:
        total_spec = sum(s["take_convs"] for s in spec["slices"].values())
        scale = min(1.0, args.limit / total_spec)

    records, manifest = [], {"variant": args.variant, "seed": args.seed,
                             "limit": args.limit, "slices": {}}
    shortfall = 0
    order = ["fine3d", "anchor", "generic3d", "general", "embodied"]  # embodied last (absorbs refill)
    for name in order:
        s = dict(spec["slices"][name])
        s["take_convs"] = int(round(s["take_convs"] * scale))
        filt = s.get("template_filter")
        fine3d_filter = set(filt) if (name == "fine3d" and filt) else None
        if name == "embodied" and shortfall:
            s["take_convs"] += shortfall
        recs, short = load_slice(name, s, rng, fine3d_filter)
        shortfall += short
        records.extend(recs)
        manifest["slices"][name] = {"take": s["take_convs"], "emitted": len(recs)}
        print(f"  {name:<10} emitted {len(recs):>9,}")
    rng.shuffle(records)

    out_path = out_dir / (f"{args.variant}{args.tag}"
                          f"{'_smoke' + str(args.limit) if args.limit else ''}.jsonl")
    h = hashlib.md5()
    with open(out_path, "w") as fh:
        for r in records:
            line = json.dumps(r, ensure_ascii=False)
            h.update(line.encode())
            fh.write(line + "\n")
    manifest["total"] = len(records)
    manifest["md5"] = h.hexdigest()[:12]

    # self-check: placeholder contract + path sample on the EMITTED file
    import os
    bad_ph = miss = 0
    lines = open(out_path).readlines()
    for l in lines:
        d = json.loads(l)
        nph = sum(c["value"].count("<image>") for c in d["conversations"] if c["from"] == "human")
        if nph != len(d["image"]):
            bad_ph += 1
    for l in rng.sample(lines, min(300, len(lines))):
        for p in json.loads(l)["image"]:
            if not os.path.exists(p):
                miss += 1
    manifest["selfcheck"] = {"placeholder_mismatch": bad_ph, "path_miss_of_300": miss}
    json.dump(manifest, open(str(out_path) + ".manifest.json", "w"), indent=1)
    status = "OK" if bad_ph == 0 and miss == 0 else "SELF-CHECK FAILED"
    print(f"{out_path.name}: {len(records):,} convs, md5 {manifest['md5']} [{status}]")
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
