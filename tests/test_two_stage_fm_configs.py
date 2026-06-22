import json
import os

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "vla_ablation_rlds")


def _load(name):
    with open(os.path.join(CFG_DIR, name)) as f:
        return json.load(f)


def test_stage1_freezes_vlm_and_trains_fmdecoder():
    c = _load("fm_dualcam_stage1.json")
    ts = c["train_setup"]
    assert ts["freeze_backbone"] is True
    assert ts["train_vision"] is False
    assert ts["train_text_embedding"] is False
    assert c["act_head"]["type"] == "FMDecoder"
    assert c["trainer"]["max_steps"] == 5000
    assert c["learning_rate"] == 2e-05
    assert c["warmup_steps"] == 1000
    assert c["resume_pretrain"] is False
    assert c["model_load_path"] is None
    assert c["norm_action"] is False


def test_stage2_full_finetune_with_resume_pretrain():
    c = _load("fm_dualcam_stage2.json")
    ts = c["train_setup"]
    assert ts["freeze_backbone"] is False
    assert ts["train_vision"] is True
    assert ts["train_text_embedding"] is True
    assert c["act_head"]["type"] == "FMDecoder"
    assert c["trainer"]["max_steps"] == 50000
    assert c["learning_rate"] == 8e-05
    assert c["warmup_steps"] == 800
    assert c["resume_pretrain"] is True
    assert c["model_load_source"] == "lightning"
    assert c["resume"] is None
    assert c["norm_action"] is False


def test_stage1_stage2_act_head_identical():
    a = _load("fm_dualcam_stage1.json")["act_head"]
    b = _load("fm_dualcam_stage2.json")["act_head"]
    assert a == b, "stage1/stage2 must share the exact same FMDecoder/DiTConfig"


def test_dualcam_and_data_match_baseline():
    for name in ("fm_dualcam_stage1.json", "fm_dualcam_stage2.json"):
        c = _load(name)
        assert c["use_hand_rgb"] is True
        assert c["train_dataset"]["type"] == "OpenVLADataset"
        assert c["train_dataset"]["data_mix"] == "libero_10_no_noops"
        assert c["fwd_pred_next_n"] == 4
