# Pixel-domain D3 robustness

This bounded operator-controlled experiment decodes the exact indexed 2 s/16-frame windows and reruns the same DINOv3-L encoder. Evaluation uses 100 real videos and up to 20 videos per generator; calibration uses the disjoint 200-real split. `jpeg_q30` applies an in-memory JPEG encode/decode and `resize_half` downsamples each input dimension by two before the standard 224 x 224 encoder transform.

## Dataset macro metrics

| Dataset | Condition | Score | AUC | AP |
|---|---|---|---:|---:|
| comgenvid | jpeg_q30 | d3_one_sided | 0.7631 | 0.7556 |
| comgenvid | jpeg_q30 | d3_raw | 0.7637 | 0.7596 |
| comgenvid | jpeg_q30 | d3_two_sided | 0.6906 | 0.6400 |
| comgenvid | reference_pixels | d3_one_sided | 0.7675 | 0.7603 |
| comgenvid | reference_pixels | d3_raw | 0.7688 | 0.7621 |
| comgenvid | reference_pixels | d3_two_sided | 0.7012 | 0.6797 |
| comgenvid | resize_half | d3_one_sided | 0.7550 | 0.7477 |
| comgenvid | resize_half | d3_raw | 0.7562 | 0.7601 |
| comgenvid | resize_half | d3_two_sided | 0.7000 | 0.6795 |
| genvideo | jpeg_q30 | d3_one_sided | 0.6759 | 0.7110 |
| genvideo | jpeg_q30 | d3_raw | 0.6753 | 0.7144 |
| genvideo | jpeg_q30 | d3_two_sided | 0.5333 | 0.5491 |
| genvideo | reference_pixels | d3_one_sided | 0.7003 | 0.7107 |
| genvideo | reference_pixels | d3_raw | 0.7003 | 0.7135 |
| genvideo | reference_pixels | d3_two_sided | 0.6022 | 0.6212 |
| genvideo | resize_half | d3_one_sided | 0.7041 | 0.7265 |
| genvideo | resize_half | d3_raw | 0.7019 | 0.7265 |
| genvideo | resize_half | d3_two_sided | 0.5694 | 0.5875 |
| videofeedback | jpeg_q30 | d3_one_sided | 0.4878 | 0.5542 |
| videofeedback | jpeg_q30 | d3_raw | 0.4885 | 0.5564 |
| videofeedback | jpeg_q30 | d3_two_sided | 0.6112 | 0.6414 |
| videofeedback | reference_pixels | d3_one_sided | 0.5079 | 0.5739 |
| videofeedback | reference_pixels | d3_raw | 0.5082 | 0.5744 |
| videofeedback | reference_pixels | d3_two_sided | 0.6013 | 0.5949 |
| videofeedback | resize_half | d3_one_sided | 0.4730 | 0.5521 |
| videofeedback | resize_half | d3_raw | 0.4745 | 0.5534 |
| videofeedback | resize_half | d3_two_sided | 0.5974 | 0.5920 |

## Stability against decoded reference

| Dataset | Condition | Score | n | Pearson r | Mean absolute change |
|---|---|---|---:|---:|---:|
| comgenvid | jpeg_q30 | d3_raw | 140 | 0.9801 | 0.1914 |
| comgenvid | resize_half | d3_raw | 140 | 0.9958 | 0.0812 |
| comgenvid | jpeg_q30 | d3_one_sided | 140 | 0.9344 | 0.0769 |
| comgenvid | resize_half | d3_one_sided | 140 | 0.9806 | 0.0382 |
| comgenvid | jpeg_q30 | d3_two_sided | 140 | 0.8107 | 0.1335 |
| comgenvid | resize_half | d3_two_sided | 140 | 0.9293 | 0.0735 |
| videofeedback | jpeg_q30 | d3_raw | 300 | 0.9648 | 0.1922 |
| videofeedback | resize_half | d3_raw | 300 | 0.9828 | 0.1273 |
| videofeedback | jpeg_q30 | d3_one_sided | 300 | 0.9072 | 0.0888 |
| videofeedback | resize_half | d3_one_sided | 300 | 0.9584 | 0.0578 |
| videofeedback | jpeg_q30 | d3_two_sided | 300 | 0.7627 | 0.1433 |
| videofeedback | resize_half | d3_two_sided | 300 | 0.8653 | 0.1014 |
| genvideo | jpeg_q30 | d3_raw | 260 | 0.9827 | 0.1889 |
| genvideo | resize_half | d3_raw | 260 | 0.9949 | 0.1009 |
| genvideo | jpeg_q30 | d3_one_sided | 260 | 0.9382 | 0.0727 |
| genvideo | resize_half | d3_one_sided | 260 | 0.9703 | 0.0442 |
| genvideo | jpeg_q30 | d3_two_sided | 260 | 0.8027 | 0.1306 |
| genvideo | resize_half | d3_two_sided | 260 | 0.9072 | 0.0807 |

## Integrity

- Failed condition rows: `0`.
- This is a declared robustness subset, not a replacement for the full strict-protocol result.
