"""
Frozen Stage 1 (GCAIN) batch inference.

Loads a trained checkpoint and runs forward on all samples in a split,
saving coarse_d.npy [N_nodes] per sample.
"""

from pathlib import Path

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import yaml


def _load_shared_assets(shared_dir: Path, device: str) -> dict:
    """Load all shared FEM assets from shared_dir and move to device."""
    mesh = np.load(shared_dir / "mesh.npz")
    nodes = torch.tensor(mesh["nodes"], dtype=torch.float32).to(device)

    A_sp = scipy.sparse.load_npz(shared_dir / "system_matrix.A.npz")
    A = torch.tensor(A_sp.toarray(), dtype=torch.float32).to(device)

    def load_lap(name: str) -> torch.Tensor:
        mat = scipy.sparse.load_npz(shared_dir / name)
        return torch.tensor(mat.toarray(), dtype=torch.float32).to(device)

    L = load_lap("graph_laplacian_full.Lap.npz")
    L0 = load_lap("graph_laplacian_full.n_Lap0.npz")
    L1 = load_lap("graph_laplacian_full.n_Lap1.npz")
    L2 = load_lap("graph_laplacian_full.n_Lap2.npz")
    L3 = load_lap("graph_laplacian_full.n_Lap3.npz")

    knn_idx = torch.tensor(
        np.load(shared_dir / "knn_idx_full.npy"), dtype=torch.long
    ).to(device)

    sens_w = torch.norm(A, dim=0)
    sens_w = sens_w / (sens_w.max() + 1e-8)

    return {
        "nodes": nodes,
        "A": A,
        "L": L, "L0": L0, "L1": L1, "L2": L2, "L3": L3,
        "knn_idx": knn_idx,
        "sens_w": sens_w,
    }


def run_stage1_inference(
    checkpoint_path: str,
    config_path: str,
    shared_dir: str,
    samples_dir: str,
    split_file: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
    activation: str = "leaky_relu",
    leaky_relu_slope: float = 0.01,
) -> None:
    """
    Load frozen GCAIN checkpoint, run forward on all samples in split,
    save coarse_d.npy [N_nodes] to output_dir/sample_id/.

    Args:
        checkpoint_path: Path to best.pth checkpoint.
        config_path: Path to the stage1 yaml config (for model hyperparams).
        shared_dir: FMT-SimGen output/shared directory.
        samples_dir: FMT-SimGen samples directory.
        split_file: Path to split txt file (one sample_id per line).
        output_dir: Root directory for bridge outputs.
        device: "cuda" or "cpu".
        batch_size: Number of samples per forward pass.
        activation: Output activation ("leaky_relu", "sigmoid", or "clamp").
        leaky_relu_slope: Slope for leaky_relu activation.
    """
    from du2vox.models.stage1.gcain import GCAIN_full

    shared_dir = Path(shared_dir)
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config for model hyperparams
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    normalize_b = data_cfg.get("normalize_b", True)

    print(f"[Stage1Inference] Loading shared assets from {shared_dir}")
    assets = _load_shared_assets(shared_dir, device)
    n_nodes = assets["nodes"].shape[0]
    n_surface = assets["A"].shape[0]
    print(f"  Mesh: {n_nodes} nodes, {n_surface} surface nodes")

    # Build model
    model = GCAIN_full(
        L=assets["L"], A=assets["A"],
        L0=assets["L0"], L1=assets["L1"], L2=assets["L2"], L3=assets["L3"],
        knn_idx=assets["knn_idx"],
        sens_w=assets["sens_w"],
        num_layer=model_cfg.get("num_layer", 6),
        feat_dim=model_cfg.get("feat_dim", 6),
    ).to(device)

    # Load checkpoint
    print(f"[Stage1Inference] Loading checkpoint {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        best_dice = ckpt.get("best_dice", float("nan"))
        print(f"  Checkpoint epoch={epoch}, best_dice@0.3={best_dice:.4f}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Activation function (must match training)
    if activation == "leaky_relu":
        def apply_activation(x):
            return F.leaky_relu(x, negative_slope=leaky_relu_slope).clamp(max=1.0)
    elif activation == "sigmoid":
        def apply_activation(x):
            return torch.sigmoid(x)
    else:
        def apply_activation(x):
            return x.clamp(0.0, 1.0)

    # Load sample IDs
    with open(split_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    # Filter to only existing samples
    sample_ids = [sid for sid in all_ids if (samples_dir / sid).exists()]
    skipped = len(all_ids) - len(sample_ids)
    if skipped:
        print(f"  Skipping {skipped} sample IDs not found in {samples_dir}")
    print(f"  Processing {len(sample_ids)} samples")

    # Preload measurements
    print("[Stage1Inference] Preloading measurements...")
    b_list = []
    for sid in sample_ids:
        b = np.load(samples_dir / sid / "measurement_b.npy").astype(np.float32)
        b = torch.tensor(b).unsqueeze(-1)  # [S, 1]
        if normalize_b:
            b_max = b.max()
            if b_max > 1e-8:
                b = b / b_max
        b_list.append(b)

    # Batch inference
    n_samples = len(sample_ids)
    n_done = 0

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            b_batch = torch.stack(b_list[start:end]).to(device)  # [B, S, 1]
            B = b_batch.shape[0]

            X0 = torch.zeros(B, n_nodes, 1, device=device)
            pred = model(X0, b_batch)
            pred = apply_activation(pred)  # [B, N, 1]
            pred = pred.clamp(0.0, 1.0)  # ensure [0,1] range
            pred_np = pred.squeeze(-1).cpu().numpy()  # [B, N]

            for i, sid in enumerate(sample_ids[start:end]):
                out_path = output_dir / sid
                out_path.mkdir(parents=True, exist_ok=True)
                np.save(out_path / "coarse_d.npy", pred_np[i])

            n_done += B
            print(f"  [{n_done}/{n_samples}] Done batch {start}:{end}")

    print(f"[Stage1Inference] Saved coarse_d.npy for {n_samples} samples to {output_dir}")
