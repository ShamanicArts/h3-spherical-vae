# Running the clean implementation

The wheel contains the circular decoder, final shifted-prediction sampler and a complete text-to-video runtime. The deployment bundle includes the wheel and a SHA-verified ComfyUI source archive. It does not import the private experiment harness or include model weights, credentials, recordings or training data.

The supported recipe is T2V, BF16 diffusion/text encoding, reviewed LoRA 1.0, the original `simple` Euler schedule, one final half-width prediction with one accepted latent column per edge, and FP16 native VAE decoding. Audio latents are saved unchanged by the correction; this example exports **silent video**, not an audio waveform. Image/video conditioning, refinement and polar repair remain separate research tasks.

## Build before launching

Use Python 3.12 for the tested fal environment, `uv` for packaging and the fal CLI configured with your own account. From a checkout:

```bash
python -m unittest discover -s tests -v
cp deployment/example-config.json /tmp/my-h3-config.json
# Set a unique run_id and output_root in that JSON before each intended run.
python scripts/build_deployment.py /tmp/my-h3-config.json /tmp/my-h3-bundle
```

Building creates no GPU job. It builds the wheel, downloads the pinned Comfy source archive and writes hashes of every runtime input. An existing archive can be supplied with `--comfy-archive`; its hash must still match. For CPU tests install NumPy and PyTorch first. GPU dependencies are declared in `deployment/fal_app.py` and installed by fal.

The example config deliberately uses the established **1024×448, 124-frame, 50-step diagnostic**. For the native comparison use **1536×672, 243 frames, 50 steps**. Dimensions are explicit: width must be divisible by 64 and height by 32; a frame count that Comfy would silently snap is rejected. No automatic reduction of resolution or steps occurs.

## Launch and invoke once

```bash
fal run /tmp/my-h3-bundle/fal_app.py::H3Spherical --auth private
```

Call the printed private synchronous endpoint with `{"run_id":"my-first-run"}` (matching the config). Use your usual fal authentication; no credentials belong in the bundle. A unique output-directory claim prevents the same run ID/output location from starting twice. A disconnected request is **not** permission to resubmit: inspect `intent.json`, `generation/progress.json`, `generation/failed.txt` and `generation/result.json` first.

Model URLs and SHA-256 values are in `deployment/models.json`. Missing weights download into the configured fal persistent cache; existing files are hash checked. Respect upstream access/license terms. The LoRA URL names the public file; its hash pins the exact reviewed weights even if that URL later changes.

The worker has a configurable exit timer (maximum one hour), no automatic request retries and a 60-second idle keepalive. These are not a dollar-budget service. Stop the ephemeral CLI/runner after collecting results and verify its terminal state in fal; persistent model/output storage is billed separately. Our validation additionally uses an external controller deadline and explicit runner termination.

## Inspect the result

Each run emits:

- Source/wheel hashes and pinned Comfy revision in `deployment.json`.
- Fresh prompt conditioning and exact final-step audit tensors.
- Control and treatment video/audio latents, including the sigma schedule.
- Ordinary and treated full ERP outputs at each requested decode context, with FFV1 masters, MP4 review copies and six PNG snapshots.
- Per-frame boundary error, hashes of every decoded master frame, model evaluation count and sampling time.
- All-frame pixel checks at the **opposite longitude**, plus an exact latent/audio check outside the accepted seam columns.

The ordinary counterfactual comes from the same real forward pass as the correction. This avoids relying on a cold-versus-reloaded model comparison. Inspect both the seam and the opposite side; a successful process or lower first/last-column error alone is not a quality verdict.

Current deployment validation status and recorded results: [release status](release-status.md).

After downloading `result.json`, the MP4s and PNG snapshots into one directory, build the repository viewer with `npm ci --ignore-scripts && npm run build:viewer`, then run `python scripts/build_run_viewer.py <download-directory>`. Serve that directory over HTTP. The synchronized A/B viewer starts with circular-128 ordinary versus final-shift output, and includes complete ERP, opposite-meridian and pole views. It does not launch model inference.
