# A0-A3 information-source audit decision

Architecture decisions below are derived from validation only.

| Gate | Pass |
| --- | --- |
| A1_coarse_state_information | GO |
| A2_terminal_latent_information | GO |
| A3_stage1_bottleneck_vs_A1 | GO |
| A3_stage2_viability | GO |
| approve_V1_local_state_decoder | GO |
| approve_latent_for_V2_or_later | GO |
| stop_V1_to_V5 | NO-GO |
| prioritize_stage1_quality | GO |

## Strict geometry

| Arm | Final Dice | Stage-II marginal | Detail cosine |
| --- | ---: | ---: | ---: |
| A0 | 0.732512 | +0.000000 | 0.000495 |
| A1 | 0.737209 | +0.004697 | 0.146292 |
| A2 | 0.747318 | +0.014806 | 0.173173 |
| A3 | 0.962357 | +0.023959 | 0.788354 |

## Direct views-on

| Arm | Final Dice | Stage-II marginal | Detail cosine |
| --- | ---: | ---: | ---: |
| VA0 | 0.732842 | +0.000330 | 0.044875 |
| VA1 | 0.742101 | +0.009589 | 0.158879 |
| VA2 | 0.747336 | +0.014824 | 0.174729 |
| VA3 | 0.959932 | +0.021534 | 0.774558 |

## Preregistered validation contrasts

| Contrast | Dice delta | Paired 95% CI |
| --- | ---: | ---: |
| A1 - A0 | +0.004697 | [+0.002956, +0.006432] |
| A2 - A1 | +0.010109 | [+0.007512, +0.012792] |
| A3 - A1 | +0.225148 | [+0.213354, +0.237513] |
| VA0 - A0 | +0.000330 | [+0.000167, +0.000497] |
| VA1 - A1 | +0.004892 | [+0.003514, +0.006278] |
| VA2 - A2 | +0.000018 | [-0.000318, +0.000341] |
| VA3 - A3 | -0.002425 | [-0.002634, -0.002218] |

Strict A0-A3 and paired views-on VA0-VA3 are reported separately in `validation_freeze.json`.

Frozen V4 was not trained; sealed confirmation was not accessed; D1-D3 and V1-V5 were not run.
