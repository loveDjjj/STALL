# Protocol terminology ledger

| Canonical term | Definition | Variants to avoid |
|---|---|---|
| STALL | Original Global spatial plus Global T1 algorithm | original STALL score without a protocol qualifier |
| K1_STALL | Reproduced STALL using the stored public fixed-seed contiguous window, VATEX component percentiles, and no second video-level CDF | K1_G when referring to the original paper pipeline |
| K1_G | Global branch under the duration-aware causal comparison calibration | original STALL |
| K1_S | Alpha Global-Local fusion under the same K1 causal chain | clean K1 without a protocol qualifier |
| K3_G | Duration-aware multi-window Global control | STALL without a K qualifier |
| K3_S | Duration-aware Alpha-STALLED with K=3 where a valid 2 s span exists | K3 Alpha without defining short-video handling |
| AP_fake | Average precision with generated video as positive; aligned with STALL Table 1 | anomaly AP when the orientation is unstated |
| AP_real | Average precision with real video as positive; historical Alpha-STALLED endpoint | AP without an orientation label in mixed-protocol tables |
| Macro-3 | Equal-weight mean of the three dataset-level generator macros | all-benchmark average |
| All-23 | Equal-weight mean over all 23 generator metrics | Macro-3 |
| calibration real | Real-only videos used to fit whitening/CDF statistics, excluded from Alpha evaluation | test real |
| evaluation real | Authentic clips used only for final metrics | calibration set |
