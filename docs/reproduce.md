# Reproducing the decoder comparison

Use the raw ComfyUI H3 visual VAE interface from revision
`12d5279438bfefc058a269eae805ceab6047777f`. Bring the model environment and weights separately.

1. Generate once and save the final **video-only** latent in the raw VAE's native normalization. Keep the original conditioning, audio latent and settings.
2. Decode that exact tensor with context 0, 128 and 384, keeping dtype, decoder configuration and export settings identical. Context 0 is the native-width control.
3. Allocate an output buffer at the extended output width when using the [buffer example](../examples/decode_saved_latent.py). At 1536 px, the three decoded widths before cropping are 1536, 1792 and 2304 px. The final width is always 1536 px.
4. Save lossless PNG frames before delivery compression. Compare identical frame indices, yaw, pitch and field of view. Review complete ERP video and temporal playback too.
5. Record the latent hash, weight hashes, actual model evaluations, dimensions, timings and output hashes. The original runs' selected records are in [evidence.json](../viewer/evidence.json).

The temple and forest each used 243 frames, 24 fps, 1536 x 672 and 50 ordinary sampling evaluations; reviewed 360 LoRA strength 1.0. H3 weights were BF16 and the visual VAE FP16. The comparison PNGs in this repo show temple frame 97 with yaw 180 degrees, pitch 0 and horizontal/vertical FOV 20 degrees.

Exact model, VAE and LoRA SHA-256 hashes, prompts and seeds are retained in the evidence JSON. A matching seed on another backend is not sufficient to recover the identical latent. Latents and model weights are not distributed here, so this repository supports repeating the comparison method rather than bitwise regeneration from a clean clone.

## Measurement

For each RGB frame on a 0-255 scale:

```python
edge_mae = abs(frame[:, 0, :].astype(float)
               - frame[:, -1, :].astype(float)).mean()
```

Average across every decoded frame. A decrease indicates lower boundary pixel mismatch. Blur can also lower this value; inspect local detail and geometry rather than selecting a variant by MAE alone. Record temporal change separately; unaligned frame differences are not flicker measurements.

## Validation boundary

The original GPU experiments validate wrapped decoding with the same coordinate plan and crop. The extracted package is checked with NumPy/PyTorch and fake raw-decoder contracts, including output-buffer handling. A real-weight package smoke test is still outstanding. No paid computation is started by these instructions or scripts.
