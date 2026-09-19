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

## Development-test results

These results were evaluated only after the validation checkpoint freeze and did not affect any gate decision.

### Strict geometry

| Arm | Final Dice | HD95 | Localization | MSE | rL2 | Stage-II marginal | Detail cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.716529 | 2.173467 | 1.352125 | 0.00663789 | 1.130143 | -0.000000 | 0.000957 |
| A1 | 0.722548 | 2.015701 | 1.261592 | 0.00680329 | 1.145395 | +0.006018 | 0.157982 |
| A2 | 0.730485 | 1.833319 | 1.189697 | 0.00692143 | 1.154513 | +0.013955 | 0.172997 |
| A3 | 0.961147 | 0.200001 | 0.076529 | 0.000370761 | 0.250752 | +0.025457 | 0.788937 |

### Direct views-on

| Arm | Final Dice | HD95 | Localization | MSE | rL2 | Stage-II marginal | Detail cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VA0 | 0.717047 | 2.173639 | 1.343358 | 0.00663517 | 1.129988 | +0.000518 | 0.047950 |
| VA1 | 0.727428 | 2.004148 | 1.249922 | 0.0067929 | 1.144114 | +0.010898 | 0.171194 |
| VA2 | 0.730547 | 1.831623 | 1.173655 | 0.00693215 | 1.155568 | +0.014017 | 0.174185 |
| VA3 | 0.958839 | 0.200001 | 0.087377 | 0.000394492 | 0.258987 | +0.023149 | 0.776187 |

### Paired development-test contrasts

| Contrast | Dice delta | Paired 95% CI |
| --- | ---: | ---: |
| A1 - A0 | +0.006018 | [+0.004403, +0.007646] |
| A2 - A1 | +0.007937 | [+0.005096, +0.010759] |
| A3 - A1 | +0.238599 | [+0.225084, +0.252195] |
| VA0 - A0 | +0.000518 | [+0.000313, +0.000738] |
| VA1 - A1 | +0.004880 | [+0.003652, +0.006153] |
| VA2 - A2 | +0.000062 | [-0.000269, +0.000392] |
| VA3 - A3 | -0.002308 | [-0.002519, -0.002100] |

Strict A0-A3 and paired views-on VA0-VA3 are reported separately in `final_go_no_go.json`.

Frozen V4 was not trained; sealed confirmation was not accessed; D1-D3 and V1-V5 were not run.
