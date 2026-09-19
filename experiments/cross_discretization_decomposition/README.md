# Cross-discretization support-space decomposition

This experiment tests, without training a Stage-2 model, whether the binary
high-resolution source-support error separates into a component representable by
the complete 20k-node P1 FEM space and its mass-orthogonal complement.

The primary domain is fixed for every case: 0.2 mm GT voxel centers contained in
the complete FEM tetrahedral domain.  It is independent of Stage-1 predictions,
ROIs, and ground truth.  The primary target is
`(gt_voxels > 0.05).astype(float64)`, matching the binary Stage-1 training target.

Run from the repository root:

```bash
uv run python experiments/cross_discretization_decomposition/run_analysis.py \
  --max-samples 5 --overwrite

uv run python experiments/cross_discretization_decomposition/run_analysis.py
```

All generated operators, tables, figures, and reports remain below `artifacts/` in
this directory.  The script never modifies checkpoints, bridge outputs,
precomputed Stage-2 data, or dataset splits.

Experiment F is intentionally skipped: the available alternative mesh directories
are not verified coarse/default/fine discretizations of this same simulation
dataset and do not have matched Stage-1 reconstructions.
