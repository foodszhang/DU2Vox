# Final dual-space go/no-go

All architecture choices below were made on validation300 before the final development-test300 evaluation. Sealed confirmation was not accessed.

1. **Q1:** Yes. The validation-frozen final method is `strong_sequential_concat_B4` and the matched chain is `0.617379 -> 0.716529 -> 0.730485` through separated coarse and complementary refinement.
2. **Q2:** The separate retrained-Stage1 Q-only arm reaches only 0.645820 because hard-Q cannot correct FEM-representable inverse error.
3. **Q3:** V4 coarse correction contributes +0.099150 Dice in the matched chain.
4. **Q4:** The validation-selected full model contributes +0.013955 over coarse-only.
5. **Q5:** CST does not significantly exceed the parameter-matched concat baseline; val paired CI is [-0.0003271397578672959, 0.0011248796405671645] and development difference is +0.000988.
6. **Q6:** Hc+Delta-H versus Hc-only has validation difference +0.000581 with CI [-2.428069579085402e-05, 0.0011739561766482502]; the Delta-H claim is retained only when this evidence is positive.
7. **Q7:** Frozen coarse is retained when separated continuation fails to improve validation; unrestricted joint training is rejected if it reproduces FEM drift.
8. **Q8:** Yes. Mean coarse preservation relative L2 is 8.773e-09, with leakage 4.501e-18.
9. **Q9:** Detail cosine=0.177899, HD95=1.846463 mm, localization=1.193214 mm.
10. **Q10:** Approximation-space separation is supported. CST is downgraded; responsibility-preserving routing is supported as a safety contract, not a performance-improving co-training claim.

*The Q-only row uses the separately frozen retrained-Stage1 protocol and is not a matched B4/CST comparison.*
