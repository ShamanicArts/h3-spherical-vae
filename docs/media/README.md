# Walkthrough excerpt

`h3-seam-walkthrough.mp4`: H.264, 1412 x 736, 60 fps, 24.5 seconds, no audio, fast-start metadata. CRF 18. The 56 px header is an editorial label; the underlying view is cropped without rescaling.

A is native decoding; B is circular decoding with 128 px context per side. Source frames 480-1949 are retained at original speed. Switches are based on the displayed decoder label in the recording, checked immediately before and after each transition. The original browser chrome, tabs, sidebars and desktop are outside the crop.

This is a moving screen recording, not a synchronized fixed-frame comparison. See the PDF and stills for those comparisons. The source recording is not distributed. Reproduce the edit with Python 3.11+ and FFmpeg using `scripts/edit_walkthrough.py SOURCE.mp4`; the source hash is checked. `h3-seam-walkthrough.json` records crop, timing and output hash.
