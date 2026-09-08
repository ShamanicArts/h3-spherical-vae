# H3 circular VAE decoding

A small horizontal-context adapter for decoding equirectangular video with MiniMax H3. In matched 50-step forest and temple examples, it reduces the visible left/right join. Every comparison decodes the **same sampled video latent** with native, 128px and 384px context.

[Interactive comparison](viewer/index.html) · [Technique](docs/technique.md) · [Evidence](docs/evidence.md) · [Refinement test](docs/refinement.md)

```text
H3 final video latent
  → [right-hand columns | original latent | left-hand columns]
  → native H3 VAE decode
  → crop added margins
  → complete ERP video at the original dimensions
```

At H3's16× spatial compression,128px context means8 latent columns per side;384px means24. Context changes the native decoder's available neighborhood **and its tile layout**. This implementation does not claim a boundary-only internal change.

```python
from circular_decode import decode_circular

# Raw ComfyUI MiniMaxH3VideoVAE; video-only BCTHW latent on its device/dtype.
pixels = decode_circular(vae, video_latent, context_pixels=128)
```

Supply the model environment and weights separately. The adapter was extracted from experiments using ComfyUI revision `12d5279438bfefc058a269eae805ceab6047777f`. It does not load weights or call a cloud service. See the technique document for memory/output-buffer handling and limitations.

## What is established

Two10.125-second,1536×672,243-frame24fps examples,50 diffusion steps, BF16 H3 and reviewed360 LoRA1.0; native visual VAE FP16. The author finds a noticeable seam improvement on both, particularly the temple. Mean boundary mismatch falls approximately28–31%; this is a diagnostic, not a perceptual quality percentage. Higher-resolution refinement and upscaling remain to be validated.

Run geometry/adapter checks with NumPy and, optionally, PyTorch:

```sh
python -m unittest discover -s tests -v
```

`spherical_context.py` also contains pole-coordinate experiments; `periodic_tiles.py` contains an alternative tile-assembly experiment. Neither is part of the default horizontal wrapper or the demonstrated50-step intervention. Model-wide longitude consistency and polar correctness are separate questions.

This repository is a selected standalone extraction. Model weights and model implementations retain their upstream licenses and are not included. The viewer can be served as static files; see [viewer/README.md](viewer/README.md).
