# Final shifted prediction report card

[Read or download the five-page PDF](../H3-final-prediction-report-card.pdf) · [Algorithm and limitations](../final-prediction.md) · [Settings and image hashes](evidence.json)

The report separates the existing circular decoder from the incremental sampling change. Native-resolution temple and forest examples use 50 steps. The timing comparison uses 100 steps at 1024 x 448. Comparisons keep decoder choice, frame and projection matched.

![100-step timing report](preview.png)

All comparison images in `assets/` are PNGs. Full ERP images are copied from verified lossless snapshots; perspective images are projected from those same snapshots using the exact FFmpeg filters in `evidence.json`. They are magnifications of existing detail, with no sharpening or generated enhancement. Image SHA-256 values and source PNG hashes are included. The PDF embeds these image pixels losslessly.

To rebuild with Python, ReportLab, NumPy and pypdf available:

```sh
python scripts/build_report_card.py
```

The builder verifies the curated assets and checks that PDF recompression preserves every embedded image's decoded bytes. It performs no inference, cloud provisioning or network requests. The report now includes opposite-longitude controls at 75 and 20 degrees. The middle half of each selected ERP frame is pixel-identical between control and final shift. License status is unchanged.
