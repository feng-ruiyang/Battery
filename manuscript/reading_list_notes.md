# Reading List Notes for SSCI 2027 Draft

Source file: `/Users/fry/Downloads/2026.7.3.pages`

The Pages document was parsed through its preview image and embedded `.iwa`
strings. Some entries are only partially recoverable, so only verified papers
should be cited in the manuscript.

## Verified and Usable Now

1. Severson et al., 2019, *Nature Energy*
   - Title: "Data-driven prediction of battery cycle life before capacity degradation"
   - DOI: `10.1038/s41560-019-0356-8`
   - Use in paper: dataset origin and early-cycle battery lifetime prediction.
   - Difference to our work: predicts cycle life / classifies lifetime from early
     discharge voltage features; does not study autoregressive probabilistic SoH
     rollout or teacher-forcing mismatch.

2. Parimala et al., 2026, *Scientific Reports*
   - Title: "Uncertainty aware hybrid learning framework for fast and safe
     charging of lithium-ion batteries using multi-fidelity observers"
   - URL from reading list: `https://www.nature.com/articles/s41598-025-31976-1`
   - Use in paper: uncertainty-aware battery-state estimation and PICP/MPIW
     evaluation under fast charging.
   - Difference to our work: focuses on SoC estimation and fast-charging control
     with multi-fidelity observer features and MLP/NLL uncertainty; it is not a
     long-horizon autoregressive SoH rollout study.

3. Pollo et al., 2024, arXiv
   - Title: "Coupling Neural Networks and Physics Equations For Li-Ion Battery
     State-of-Charge Prediction"
   - arXiv: `2412.16724`
   - Use in paper: physics-informed neural-network battery state prediction and
     variable horizon SoC forecasting.
   - Difference to our work: studies SoC prediction with physics equations and
     two cascaded branches; uncertainty calibration and VBLL stochastic rollout
     are not the main focus.

4. Wang et al., 2026, *IEEE Transactions on Pattern Analysis and Machine
   Intelligence*
   - Title: "A Review of Uncertainty Representation and Quantification in Neural
     Networks"
   - DOI: `10.1109/TPAMI.2025.3626645`
   - Use in paper: general uncertainty quantification background, especially
     aleatoric/epistemic uncertainty, Bayesian neural networks, ensembles, and
     evidential/second-order uncertainty representations.
   - Difference to our work: general UQ review, not battery-specific and not a
   rollout training method.

5. Li et al., 2025, *Batteries*
   - Title: "A Physics-Guided Transformer for Robust State of Charge Estimation
     in Aging Lithium-Ion Batteries"
   - DOI: `10.3390/batteries11120446`
   - Use in paper: Transformer-based battery state estimation with physics
     information embedded into the architecture.
   - Difference to our work: estimates SoC, uses MSE and physics-guided
     embeddings; does not address stochastic SoH rollout calibration.

6. Jia et al., 2023, *Energy*
   - Title: "State of health prediction of lithium-ion batteries based on
     bidirectional gated recurrent unit and transformer"
   - DOI: `10.1016/j.energy.2023.129401`
   - Use in paper: battery SoH prediction with a BiGRU-Transformer model and
     long-term temporal dependencies.
   - Difference to our work: deterministic SoH prediction with error metrics;
     no VBLL uncertainty head and no rollout-aligned probabilistic training.

7. Hu et al., 2025, *Nature Communications*
   - Title: "Early prediction of lithium-ion battery degradation with a
     generative pre-trained transformer"
   - DOI: `10.1038/s41467-025-66819-0`
   - Use in paper: Transformer/GPT-style autoregressive prediction of battery
     operating features over the lifecycle.
   - Difference to our work: predicts future charging data and then estimates
     degradation; not a direct uncertainty-calibrated SoH rollout comparison.

8. Zhang et al., 2026, *Energies*
   - Title: "BatteryMoE: An Improved Sparse Mixture-of-Experts Transformer for
     Battery SOH Estimation and RUL Prediction"
   - DOI: `10.3390/en19112554`
   - Use in paper: Transformer-style SOH/RUL prediction with iterative rollout
     during RUL testing.
   - Difference to our work: focuses on architecture and RUL prediction; our
     short paper isolates teacher forcing versus stochastic rollout-aligned
     uncertainty learning under a fixed VBLL architecture.

## Partially Identified; Do Not Cite Until Confirmed

1. MDPI Batteries entry
   - URL fragment from Pages: `https://www.mdpi.com/2313-0105/11/12/446`
   - Notes in Pages: "physical-inform", "SoC", possibly offline inventory/cycle
     count and MSE. The exact title and authors were not recovered. Verify before
     citing.

2. ScienceDirect entry
   - URL fragment from Pages: `https://www.sciencedirect.com/science/article/pii/S2405896325030411`
   - Notes in Pages: "evid", "Normal-Inverse-Gamma", "solid work". The exact
     title and authors were not recovered. Verify before citing.

3. IEEE entry
   - Fragment from Pages: `ieeexplore... docu /1073270`
   - Notes in Pages: GRU + attention, SoC proportional, Kalman filter,
     time-accumulated errors. The document number seems truncated. Verify before
     citing.

4. SimMTM / masked time-series modeling
   - User previously provided the NeurIPS 2023 SimMTM paper PDF.
   - Use only if the final paper discusses masked encoder pretraining. It is not
     needed for the short paper unless SIM-style encoder results are included.
