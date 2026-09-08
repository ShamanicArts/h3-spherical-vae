# v0.1.0 preparation status

Prepared as an experimental source release. An original-code license has not yet been selected.

Verified locally:

- All 16 geometry, PyTorch and raw-decoder contract tests pass, including CPU output-buffer use.
- The wheel builds, installs in an isolated target and imports outside the checkout. Native-width, 128 px and 384 px fake-decoder checks pass from the installed wheel.
- Viewer bundle builds and inline JavaScript parses. The camera-input scene-selection defect is corrected; interactive browser playback was not revalidated for this release candidate.
- Selected source files were checked for private workspace paths and credential patterns. Model weights, run receipts, private session transcripts and video media are excluded from the source archive.

The measured real-GPU results come from the original experimental harness. The extracted callable still needs a real-weight smoke test. Alternative periodic tiling, pole handling, seam-local noise/blur and higher-resolution refinement are not established features of the release.

The visual PDF and selected comparison PNGs are included in the repository. The original walkthrough remains a separate presentation asset. Source history stays in the clean project, with Jujutsu as the local VCS. Packaging does not start cloud generation or upload to a package index.
