#!/usr/bin/env python3
"""Emit the machine-readable and human-readable P0-1 projection contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from du2vox.bridge.canonical_cross_discretization import CanonicalCrossDiscretization
from du2vox.models.stage2.complement_voxel_detail import ExactVoxelComplement


def relative(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(second), 1e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("experiments/cross_discretization_decomposition/artifacts/operator_cache"),
    )
    parser.add_argument(
        "--json-output", type=Path,
        default=Path("diagnosis/p0_weighted_l2_projection_contract.json"),
    )
    parser.add_argument(
        "--markdown-output", type=Path,
        default=Path("diagnosis/p0_weighted_l2_projection_contract.md"),
    )
    args = parser.parse_args()
    canonical = CanonicalCrossDiscretization(args.cache, factorize=True)
    complement = ExactVoxelComplement(
        canonical.p, quadrature_weight=canonical.quadrature_weight
    )
    rng = np.random.default_rng(20260902)
    x = rng.standard_normal(complement.n_fem_nodes)
    z = rng.standard_normal(complement.n_voxels)
    y = rng.standard_normal(complement.n_voxels)
    ihx = canonical.prolong(x)
    phz = complement.project_numpy(z)
    qz = complement.apply_numpy(z)
    lhs = canonical.inner_product(z, complement.project_numpy(y))
    rhs = canonical.inner_product(complement.project_numpy(z), y)
    total_energy = canonical.weighted_energy(z)
    split_energy = canonical.weighted_energy(phz) + canonical.weighted_energy(qz)
    final = ihx + qz
    errors = {
        "pi_h_i_h_relative_l2": relative(canonical.project_coefficients(ihx), x),
        "p_h_idempotence_relative_l2": relative(complement.project_numpy(phz), phz),
        "q_idempotence_relative_l2": relative(complement.apply_numpy(qz), qz),
        "p_h_weighted_self_adjoint_relative": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300),
        "p_h_q_relative_l2": np.linalg.norm(complement.project_numpy(qz)) / max(np.linalg.norm(qz), 1e-300),
        "pi_h_q_relative_l2": np.linalg.norm(canonical.project_coefficients(qz)) / max(np.linalg.norm(qz), 1e-300),
        "weighted_pythagorean_relative": abs(total_energy - split_energy) / max(total_energy, 1e-300),
        "final_coarse_preservation_relative_l2": relative(canonical.project_coefficients(final), x),
    }
    if max(errors.values()) >= 1e-12:
        raise RuntimeError(f"Weighted projection contract failed: {errors}")
    result = {
        "status": "passed",
        "formula": {
            "pi_h": "(P^T W P)^-1 P^T W",
            "i_h": "P",
            "p_h": "I_h Pi_h",
            "q": "I - P_h",
            "w": f"{canonical.quadrature_weight:.17g} I",
            "production_equivalence": "W=wI, so scalar w cancels and Pi_h=(P^T P)^-1 P^T",
        },
        "scope": canonical.inner_product_scope,
        "domain": {
            "definition": canonical.metadata["definition"],
            "grid_shape": list(canonical.operator.grid_shape),
            "spacing_mm": canonical.operator.spacing_mm,
            "valid_voxels": complement.n_voxels,
            "fem_nodes": complement.n_fem_nodes,
            "quadrature_weight_mm3": canonical.quadrature_weight,
        },
        "operator_cache_sha256": canonical.cache_hashes,
        "errors": errors,
        "tolerance": 1e-12,
        "limitations": [
            "Certified only for this fixed uniformly weighted sampled domain.",
            "This is not a continuous-domain orthogonality claim.",
            "A nonuniform grid requires retaining W explicitly in production.",
        ],
        "confirmation_data_used": False,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    rows = "\n".join(f"| `{key}` | `{value:.6e}` |" for key, value in errors.items())
    hashes = "\n".join(f"- `{name}`: `{digest}`" for name, digest in canonical.cache_hashes.items())
    args.markdown_output.write_text(
        f"""# P0-1 Weighted L2 Projection Contract

## Result

**Passed.** The unchanged production projection is the weighted sampled-L2 projection

```text
Pi_h = (P^T W P)^-1 P^T W
I_h  = P
P_h  = I_h Pi_h
Q    = I - P_h
```

On the certified uniform 0.2-mm domain, `W = {canonical.quadrature_weight:.17g} I`.
The scalar weight cancels exactly, so the existing production solve
`(P^T P)^-1 P^T` is the same operator. No second projection was introduced.

## Certified domain and inner product

- Scope: {canonical.inner_product_scope}.
- Grid shape: `{canonical.operator.grid_shape}`.
- Valid voxel centers: `{complement.n_voxels}`; FEM nodes: `{complement.n_fem_nodes}`.
- Inner product: `<u,v>_W = u^T W v` with voxel volume in mm^3.

This contract is limited to the fixed sampled domain. It is not a continuous-domain
orthogonality claim, and a nonuniform sampled domain would require explicit `W`.

## Numerical audit

| Identity | Relative error |
| --- | ---: |
{rows}

All FP64 core errors are below `1e-12`.

## Cache identity

{hashes}

No confirmation data or confirmation inference was used.
"""
    )


if __name__ == "__main__":
    main()
