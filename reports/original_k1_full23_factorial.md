# Original-window K1 x Global-Local factorial evaluation

K1 uses the public STALL fixed-seed contiguous 1s/2s window stored in the source indexes. K3 uses deterministic beginning/middle/end coverage. K1_STALL applies the original VATEX component percentiles and no second video-level CDF. The four causal cells K1_G/K1_S/K3_G/K3_S share ncustom duration-aware real-only calibration, the same video identities, and the same score direction.

## Original-paper-size standalone STALL reproduction

This row uses the paper's per-generator fake counts and real-pool caps. It may use target real videos that belong to Alpha's Local calibration, which is valid for the VATEX-calibrated Global-only STALL reproduction but makes the row ineligible for a causal comparison against Alpha. The local ComGenVid index contains 1,698 rather than the paper's 1,700 real files, so both ComGenVid pairs use 1,698.

| Scope | Reproduced K1 STALL AUC/AP_fake/AP_real | Published STALL AUC/AP_fake |
|---|---:|---:|
| comgenvid | 0.8532/0.8459/0.8553 | 0.85/0.86 |
| videofeedback | 0.8459/0.8260/0.8607 | 0.83/0.85 |
| genvideo | 0.8119/0.7996/0.8067 | 0.80/0.80 |
| All-23 | 0.8318/0.8163/0.8368 | 0.82/0.82 |

## full_coverage

Values are AUC/AP_fake/AP_real. AP_fake follows the original STALL paper; AP_real is retained for continuity with this repository's historical tables.

| Config | ComGenVid | VideoFeedback | GenVideo | Macro-3 | All-23 |
|---|---:|---:|---:|---:|---:|
| K1_STALL | 0.8510/0.8406/0.8546 | 0.8488/0.8303/0.8621 | 0.8007/0.7855/0.7966 | 0.8335/0.8188/0.8377 | 0.8281/0.8117/0.8330 |
| K1_G | 0.8511/0.8390/0.8537 | 0.8489/0.8260/0.8615 | 0.8082/0.7894/0.8053 | 0.8360/0.8181/0.8402 | 0.8314/0.8112/0.8364 |
| K1_S | 0.8910/0.8828/0.8969 | 0.8662/0.8502/0.8759 | 0.8418/0.8296/0.8336 | 0.8663/0.8542/0.8688 | 0.8577/0.8441/0.8593 |
| K3_G | 0.8628/0.8449/0.8709 | 0.8478/0.8248/0.8608 | 0.8208/0.8085/0.8111 | 0.8438/0.8260/0.8476 | 0.8373/0.8194/0.8401 |
| K3_S | 0.9045/0.8973/0.9117 | 0.8605/0.8418/0.8710 | 0.8550/0.8495/0.8417 | 0.8733/0.8629/0.8748 | 0.8619/0.8500/0.8618 |

## paper_count_fake_cohort

Values are AUC/AP_fake/AP_real. AP_fake follows the original STALL paper; AP_real is retained for continuity with this repository's historical tables.

| Config | ComGenVid | VideoFeedback | GenVideo | Macro-3 | All-23 |
|---|---:|---:|---:|---:|---:|
| K1_STALL | 0.8510/0.8406/0.8546 | 0.8488/0.8302/0.8625 | 0.8015/0.7858/0.7983 | 0.8338/0.8189/0.8384 | 0.8284/0.8118/0.8339 |
| K1_G | 0.8511/0.8390/0.8537 | 0.8490/0.8259/0.8619 | 0.8088/0.7893/0.8070 | 0.8363/0.8181/0.8409 | 0.8317/0.8111/0.8373 |
| K1_S | 0.8910/0.8828/0.8969 | 0.8660/0.8498/0.8758 | 0.8420/0.8288/0.8354 | 0.8663/0.8538/0.8694 | 0.8577/0.8436/0.8601 |
| K3_G | 0.8628/0.8449/0.8709 | 0.8478/0.8247/0.8612 | 0.8214/0.8084/0.8128 | 0.8440/0.8260/0.8483 | 0.8376/0.8193/0.8410 |
| K3_S | 0.9045/0.8973/0.9117 | 0.8604/0.8414/0.8709 | 0.8552/0.8488/0.8436 | 0.8734/0.8625/0.8754 | 0.8620/0.8495/0.8626 |

## balanced_seed42

Values are AUC/AP_fake/AP_real. AP_fake follows the original STALL paper; AP_real is retained for continuity with this repository's historical tables.

| Config | ComGenVid | VideoFeedback | GenVideo | Macro-3 | All-23 |
|---|---:|---:|---:|---:|---:|
| K1_STALL | 0.8480/0.8382/0.8510 | 0.8492/0.8295/0.8635 | 0.7945/0.7770/0.7897 | 0.8306/0.8149/0.8347 | 0.8253/0.8075/0.8303 |
| K1_G | 0.8481/0.8367/0.8503 | 0.8493/0.8252/0.8629 | 0.8022/0.7815/0.7986 | 0.8332/0.8144/0.8373 | 0.8287/0.8072/0.8339 |
| K1_S | 0.8880/0.8806/0.8932 | 0.8684/0.8524/0.8788 | 0.8383/0.8237/0.8305 | 0.8649/0.8523/0.8675 | 0.8570/0.8424/0.8590 |
| K3_G | 0.8620/0.8446/0.8697 | 0.8481/0.8241/0.8621 | 0.8109/0.7912/0.8043 | 0.8404/0.8200/0.8454 | 0.8332/0.8116/0.8376 |
| K3_S | 0.9035/0.8967/0.9109 | 0.8623/0.8436/0.8736 | 0.8504/0.8421/0.8378 | 0.8721/0.8608/0.8741 | 0.8607/0.8475/0.8613 |

## Paired bootstrap

| Contrast | Metric | Mean AP delta | 95% CI |
|---|---|---:|---:|
| duration_calibration_at_K1 | ap_real | +0.0025 | [+0.0021, +0.0029] |
| duration_calibration_at_K1 | ap_fake | -0.0005 | [-0.0013, +0.0002] |
| local_at_K1 | ap_real | +0.0301 | [+0.0269, +0.0335] |
| local_at_K1 | ap_fake | +0.0377 | [+0.0334, +0.0425] |
| local_at_K3 | ap_real | +0.0286 | [+0.0252, +0.0318] |
| local_at_K3 | ap_fake | +0.0406 | [+0.0351, +0.0461] |
| window_for_Alpha | ap_real | +0.0066 | [+0.0044, +0.0090] |
| window_for_Alpha | ap_fake | +0.0085 | [+0.0057, +0.0112] |
| total_K3_Alpha_vs_K1_Global | ap_real | +0.0367 | [+0.0329, +0.0405] |
| total_K3_Alpha_vs_K1_Global | ap_fake | +0.0462 | [+0.0410, +0.0516] |
| total_K3_Alpha_vs_original_STALL | ap_real | +0.0392 | [+0.0354, +0.0432] |
| total_K3_Alpha_vs_original_STALL | ap_fake | +0.0457 | [+0.0404, +0.0511] |

## Selection-seed sensitivity

- duration_calibration_at_K1 AP_real: mean +0.0024, std 0.0001, range [+0.0021, +0.0027].
- duration_calibration_at_K1 AP_fake: mean -0.0007, std 0.0003, range [-0.0012, +0.0001].
- local_at_K1 AP_real: mean +0.0293, std 0.0007, range [+0.0277, +0.0309].
- local_at_K1 AP_fake: mean +0.0372, std 0.0014, range [+0.0348, +0.0417].
- local_at_K3 AP_real: mean +0.0278, std 0.0007, range [+0.0262, +0.0294].
- local_at_K3 AP_fake: mean +0.0378, std 0.0014, range [+0.0350, +0.0425].
- window_for_Alpha AP_real: mean +0.0059, std 0.0006, range [+0.0048, +0.0080].
- window_for_Alpha AP_fake: mean +0.0083, std 0.0008, range [+0.0064, +0.0100].
- total_K3_Alpha_vs_K1_Global AP_real: mean +0.0352, std 0.0009, range [+0.0328, +0.0373].
- total_K3_Alpha_vs_K1_Global AP_fake: mean +0.0455, std 0.0015, range [+0.0419, +0.0495].
- total_K3_Alpha_vs_original_STALL AP_real: mean +0.0376, std 0.0010, range [+0.0350, +0.0399].
- total_K3_Alpha_vs_original_STALL AP_fake: mean +0.0448, std 0.0015, range [+0.0410, +0.0486].
