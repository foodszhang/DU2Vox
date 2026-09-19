from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

from scripts.freeze_stage1_hard_q_selection import main as freeze_main
from scripts.train_complement_voxel_detail import FullDomainEngine


ROOT = Path(__file__).resolve().parents[1]
NO_VIEWS = ROOT / "configs/stage2/stage1_hard_q_seed20260901.yaml"
VIEWS = ROOT / "configs/stage2/stage1_hard_q_views_seed20260901.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_stage1_hard_q_configs_have_no_v4_dependency() -> None:
    forbidden = ("v4", "terminal", "corrected_state", "frozen_backbone")
    for path in (NO_VIEWS, VIEWS):
        text = path.read_text().lower()
        assert all(token not in text for token in forbidden)
        cfg = _config(path)
        assert cfg["coarse_source"]["type"] == "stage1_bridge"
        assert cfg["coarse_source"]["expected_epoch"] == 122
        assert cfg["model"]["view_encoder"] is True
        assert cfg["model"]["train_view_encoder"] is True


def test_matched_arm_configs_differ_only_in_name_and_view_switch() -> None:
    no_views = _config(NO_VIEWS)
    views = _config(VIEWS)
    no_views["experiment"]["name"] = "matched"
    views["experiment"]["name"] = "matched"
    no_views["model"]["use_views"] = True
    assert no_views == views


class _PerturbationSensitiveEncoder:
    def sample_encoded(self, encoded, coords_world, coords_vox_norm=None):
        del coords_vox_norm
        value = torch.as_tensor(encoded, dtype=coords_world.dtype)
        features = value.expand(coords_world.shape[0], coords_world.shape[1], 32)
        return features, torch.ones(*coords_world.shape[:2], 7, dtype=torch.bool)


def _minimal_engine(use_views: bool) -> FullDomainEngine:
    engine = FullDomainEngine.__new__(FullDomainEngine)
    engine.use_views = use_views
    engine.cfg = {"model": {"view_feat_dim": 32}}
    engine.device = torch.device("cpu")
    engine.coords_world = torch.zeros(4, 3)
    engine.encoder = _PerturbationSensitiveEncoder()
    return engine


def test_no_view_slot_is_strictly_projection_insensitive() -> None:
    engine = _minimal_engine(use_views=False)
    first = engine.sample_view_features(torch.tensor(1.0), start=0, end=4, dtype=torch.float32)
    perturbed = engine.sample_view_features(
        torch.tensor(999.0), start=0, end=4, dtype=torch.float32
    )
    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(first, perturbed)


def test_direct_view_slot_responds_to_projection_perturbation() -> None:
    engine = _minimal_engine(use_views=True)
    first = engine.sample_view_features(torch.tensor(1.0), start=0, end=4, dtype=torch.float32)
    perturbed = engine.sample_view_features(
        torch.tensor(2.0), start=0, end=4, dtype=torch.float32
    )
    assert not torch.equal(first, perturbed)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_result(
    tmp_path: Path,
    *,
    name: str,
    use_views: bool,
    lr: float,
    epochs: int,
    dice: float,
) -> Path:
    checkpoint = tmp_path / f"{name}.pth"
    checkpoint.write_bytes(name.encode())
    cfg = _config(VIEWS if use_views else NO_VIEWS)
    cfg["training"]["lr"] = lr
    cfg["training"]["epochs"] = epochs
    config = tmp_path / f"{name}.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = {
        "split": "val",
        "n_samples": 300,
        "mode": "hard",
        "coarse_source": "stage1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "config": str(config),
        "config_sha256": _sha(config),
        "training_lr": lr,
        "training_epochs": epochs,
        "use_views": use_views,
        "initial_decoder_sha256": "same-decoder",
        "initial_encoder_sha256": "same-encoder",
        "training_order_sha256": ["same-order"] * 5,
        "parameter_counts": {"total": 645090, "trainable": 645090},
        "summary": {"final_dice": dice},
        "per_sample": [
            {"sample_id": f"sample_{index:04d}", "final_dice": dice}
            for index in range(300)
        ],
    }
    output = tmp_path / f"{name}.json"
    output.write_text(json.dumps(result))
    return output


def test_validation_selector_enforces_tuning_and_writes_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    no_views = _validation_result(
        tmp_path, name="no_views", use_views=False, lr=1e-4, epochs=5, dice=0.60
    )
    views = _validation_result(
        tmp_path, name="views", use_views=True, lr=1e-4, epochs=5, dice=0.61
    )
    low = _validation_result(
        tmp_path, name="low", use_views=True, lr=3e-5, epochs=10, dice=0.62
    )
    high = _validation_result(
        tmp_path, name="high", use_views=True, lr=3e-4, epochs=10, dice=0.63
    )
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_stage1_hard_q_selection.py",
            "--no-views",
            str(no_views),
            "--views",
            str(views),
            "--tuning",
            str(low),
            str(high),
            "--output",
            str(receipt),
        ],
    )
    freeze_main()
    frozen = json.loads(receipt.read_text())
    assert frozen["status"] == "frozen_on_val300"
    assert frozen["selected_val_dice"] == 0.63
    assert frozen["selected_use_views"] is True
    assert len(frozen["candidates"]) == 4
    assert frozen["sealed_confirmation_accessed"] is False
