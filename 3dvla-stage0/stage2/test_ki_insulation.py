#!/usr/bin/env python3
"""Unit tests for QwenFastGR00T_KI — the 5 gates from stage2-ki-design.md §4.

Usage (starVLA env, one free GPU; test 3 needs --stage1-ckpt):
  HF_HOME=/root/angli/hf_cache PYTHONPATH=/workspace/tingting/starVLA \
    python test_ki_insulation.py --device cuda:0 \
    [--stage1-ckpt /workspace/tingting/3dvla-checkpoints/stage1_full_r30k/final_model/model.safetensors]

Gates:
  1 stop-grad     : flow-only backward -> backbone grads all None/zero
  2 vocab-insul.  : fast-only backward -> lm_head & placeholder row untouched,
                    action_embed/action_lm_head trained, backbone trained
  3 bit-equival.  : Stage-1 weights grafted -> text logits identical to a
                    standalone Qwen3-VL load of the same weights
  4 FAST roundtrip: action -> ids -> decode -> action (quantization tolerance)
  5 mask exclusiv.: fast_labels supervise exactly the k+1 slots, nothing else
"""
import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, "/workspace/tingting/starVLA")

RESULTS = []


def gate(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                fn(*a, **k)
                RESULTS.append((name, "PASS", ""))
                print(f"[PASS] {name}", flush=True)
            except AssertionError as e:
                RESULTS.append((name, "FAIL", str(e)))
                print(f"[FAIL] {name}: {e}", flush=True)
        return wrapped
    return deco


def make_batch(n=2, horizon=16):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    return [{"image": [img, img], "lang": f"pick up the object number {i}.",
             "action": np.random.uniform(-1, 1, (horizon, 7)).astype(np.float32)}
            for i in range(n)]


def grad_state(module):
    """-> (n_params_with_nonzero_grad, n_params_total)"""
    nz, total = 0, 0
    for p in module.parameters():
        total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            nz += 1
    return nz, total


@gate("1 stop-grad (flow-only)")
def test_stopgrad(model):
    model.zero_grad(set_to_none=True)
    model.w_fast, model.w_flow = 0.0, 1.0
    out = model(make_batch())
    out["action_loss"].backward()
    nz_bb, _ = grad_state(model.qwen_vl_interface)
    nz_flow, n_flow = grad_state(model.flow_action_model)
    assert nz_bb == 0, f"flow loss leaked into backbone: {nz_bb} params with grad"
    assert nz_flow > 0, "flow head received no gradient"


@gate("2 vocab insulation (fast-only)")
def test_vocab_insulation(model):
    # Qwen3-VL-4B TIES lm_head to embed_tokens, so "lm_head.weight.grad == 0"
    # is unassertable (input-side prompt-token grads land in the shared
    # tensor legitimately). The tie-agnostic insulation statement is:
    # (a) the loss has NO autograd path through the vocab logits, and
    # (b) the PLACEHOLDER row of the shared weight stays untouched (its only
    #     possible paths are the hook-discarded input row and the logits).
    from starVLA.model.framework.VLM4A.QwenFastGR00T_KI import PLACEHOLDER_ID
    model.zero_grad(set_to_none=True)
    model.w_fast, model.w_flow = 1.0, 0.0

    captured = {}
    lm_head = model.qwen_vl_interface.model.get_output_embeddings()
    hook = lm_head.register_forward_hook(
        lambda m, i, o: captured.setdefault("logits", o))
    try:
        out = model(make_batch())
    finally:
        hook.remove()
    (g,) = torch.autograd.grad(out["action_loss"], captured["logits"],
                               allow_unused=True, retain_graph=True)
    assert g is None or g.abs().sum() == 0, \
        "action CE has an autograd path through the vocab logits"

    out["action_loss"].backward()
    emb = model.qwen_vl_interface.model.get_input_embeddings()
    if emb.weight.grad is not None:
        row = emb.weight.grad[PLACEHOLDER_ID]
        assert row.abs().sum() == 0, "placeholder embedding row received gradient"
    nz_ae, _ = grad_state(model.action_embed)
    nz_ah, _ = grad_state(model.action_lm_head)
    assert nz_ae > 0 and nz_ah > 0, "KI modules received no gradient"
    nz_bb, _ = grad_state(model.qwen_vl_interface.model.model)
    assert nz_bb > 0, "backbone got NO gradient from FAST CE (KI point defeated)"


@gate("3 Stage-1 bit-equivalence")
def test_bit_equivalence(model, ckpt, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor
    sd = load_file(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys grafting stage1: {unexpected[:5]}"
    # Reference must use the SAME attention implementation as the framework —
    # FA2 and sdpa are numerically different bf16 kernel paths, and this gate
    # asserts graft correctness, not kernel equivalence.
    ref = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    prefix = "qwen_vl_interface.model."
    ref_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    ref.load_state_dict(ref_sd, strict=False)
    ref = ref.to(device).eval()

    # Separate failure modes: weights first, then logits.
    g_sd = model.qwen_vl_interface.model.state_dict()
    for k in ["model.language_model.layers.0.self_attn.q_proj.weight",
              "model.language_model.layers.35.mlp.down_proj.weight",
              "lm_head.weight"]:
        if k in g_sd:
            assert torch.equal(g_sd[k].cpu(), ref_sd[k].cpu()), f"weight mismatch after graft: {k}"
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "How far is the object from the camera, in centimeters?"}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        la = model.qwen_vl_interface.model(**inputs).logits
        lb = ref(**inputs).logits
    assert torch.equal(la, lb), \
        f"grafted logits differ from standalone load (max abs diff {(la-lb).abs().max()})"
    del ref
    torch.cuda.empty_cache()


@gate("4 FAST roundtrip")
def test_fast_roundtrip(model):
    horizon = model.action_horizon
    acts = [np.random.uniform(-0.8, 0.8, (horizon, 7)).astype(np.float32)]
    ids = model._encode_fast(acts)
    dec = model.fast_action_model.fast_tokenizer.decode(ids)
    err = np.abs(np.asarray(dec)[0][:horizon] - acts[0]).max()
    assert err < 0.15, f"FAST roundtrip error {err:.3f} exceeds quantization tolerance"


@gate("5 supervision-mask exclusivity")
def test_mask_exclusivity(model):
    from starVLA.model.framework.VLM4A.QwenFastGR00T_KI import PLACEHOLDER_ID, STOP_ID
    batch = make_batch(n=2)
    fast_ids = model._encode_fast([ex["action"] for ex in batch])
    slot_ids = [ids + [STOP_ID] for ids in fast_ids]
    qi = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[ex["image"] for ex in batch], instructions=[ex["lang"] for ex in batch])
    Lp = qi["input_ids"].shape[1]
    input_ids, attn, labels = model._append_slots(qi, slot_ids)
    model._pending_slot_ids = None
    for i, slots in enumerate(slot_ids):
        k = len(slots)
        assert (labels[i] != -100).sum() == k, "label count != slot count"
        assert (labels[i, :Lp] == -100).all(), "supervision leaked into prompt/text"
        assert (input_ids[i, Lp:Lp + k] == PLACEHOLDER_ID).all()
        assert (attn[i, Lp:Lp + k] == 1).all() and (attn[i, Lp + k:] == 0).all()
        got = labels[i, Lp:Lp + k].tolist()
        assert got == slots, "labels do not equal slot ids (input==label symmetry)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage1-ckpt", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.create({
        "framework": {"name": "QwenFastGR00T_KI",
                      "qwenvl": {"base_vlm": "Qwen/Qwen3-VL-4B-Instruct",
                                 "attn_implementation": "flash_attention_2"}},
        "datasets": {"vla_data": {}},
    })
    from starVLA.model.framework.base_framework import build_framework
    model = build_framework(cfg).to(args.device)
    model.train()

    test_stopgrad(model)
    test_vocab_insulation(model)
    test_fast_roundtrip(model)
    test_mask_exclusivity(model)
    if args.stage1_ckpt:
        model.eval()
        test_bit_equivalence(model, args.stage1_ckpt, args.device)
    else:
        print("[SKIP] 3 Stage-1 bit-equivalence (no --stage1-ckpt)")

    print("\n==== summary ====")
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name} {msg[:120]}")
    sys.exit(0 if all(s == "PASS" for _, s, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()
