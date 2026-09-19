# Final dual-space experiment provenance

This table was recorded before CST implementation or model selection. It keeps the
V4 experiment contract separate from the retrained-Stage1 hard-Q audit.

| Contract item | Final CST / matched B4 contract |
| --- | --- |
| Initial Stage I checkpoint | `runs/stage1_fmt_simgen_v2_3k_20k_balanced_v2_eval/checkpoints/best.pth` |
| Initial Stage I SHA256 | `f3c8fb07007b822312ada67a8f3325ff49a4f2344fa8ebf110e954ba9a6e73d3` |
| FEM correction | unified dual-evidence V4, three shared iterations |
| FEM config / SHA256 | `configs/stage2/unified_dual_evidence_fem_v4_2400.yaml` / `e62eaa0b48d207516cc7ee6f408e8d45e04755f4dcfadc9c20e7254dffc1d4c5` |
| FEM checkpoint / SHA256 | `runs/unified_dual_evidence_fem_v4_2400/checkpoints/best_dense_val_delta_dice.pth` (epoch 15) / `bceef34e8cf49ffcf777dadaa59514681abddf46836c2a2d7a4b2120f95d2b02` |
| Terminal latent | Frozen V4 final shared-cell hidden, 19,990 x 144, audited FP16 cache |
| Initial correction latent | Frozen V4 first shared-cell hidden, same representation; opt-in trajectory cache |
| Hard-Q | `Q = I - I_h (I_h^T I_h)^-1 I_h^T`, fixed FP64 sparse forward/backward |
| Train / validation / development-test | 2400 / 300 / 300 fixed D0 samples |
| Split SHA256 | train `870ec2c7270acdb274048e32f3d69d09e212721e2412b93a5e680248f6c3a66c`; val `43c754937c0af643616033e2162154908e0a2a2ba35e0ef882f709eeefb48598`; test `17bdd83a5b3133503df8954ba3234b11394c31fe7d39c5cffe83a999223073e3` |
| Coarse-side views | Frozen V4 seven-view optical encoder |
| Direct voxel-side views | Off initially; retained only if the preregistered validation gate passes |
| Strong decoder baseline | A2 ordinary concat, 231-D fixed input, width 160, 3 residual blocks, 193,121 parameters |
| Loss | SmoothL1 hard-Q detail + 0.1 detail MSE + 0.1 final Tversky |
| Seed | `20260901` |
| Coordinates | `mcx_trunk_local_mm`; manifest normalization to `[-1,1]`; six-frequency PE |
| GT / evaluation thresholds | `gt_voxels > 0.05`; raw reconstruction threshold `0.5` |
| Mesh / frame SHA256 | `718cb70f0b12c3c8f5e7f10d9e216295efc79ff3221602b934ba04c137f34c64` / `25b77f5d11430ebf4d9735f19534782afb56f7fe5cab525a569028b0f3c02e2b` |
| Recorded code state | Git HEAD `46f650d41d4938aa700c7e637a14e074072fce8d`, branch `exp/p0-weighted-soft-joint`, dirty pre-existing research worktree |

The retrained-Stage1 hard-Q results (`0.631170 -> 0.645820`) use a different Stage I
checkpoint and remain a separate B0--B2 responsibility audit. They are not pooled
with V4/B4/CST casewise comparisons.
