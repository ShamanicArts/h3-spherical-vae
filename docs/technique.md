# Where the intervention happens

The final denoising output is a video latent with shape B×C×T×H×W. The original decoder-only 50-step comparisons use ordinary sampling. The intervention starts after that final latent has been saved; text conditioning, noise, sampling trajectory, LoRA and audio latent are shared across decoder arms.

For a 1536 px-wide H3 output, latent width is 96. With 128 px context, prepend the rightmost 8 columns and append the leftmost 8, producing 112 columns. Native decoding produces 1792 px; cropping 128 px from both sides restores 1536 px. With 384 px, use 24 columns on each side, then crop 384 px.

The copied context places the genuine spherical neighbors beyond each boundary. Every temporal slice receives the same mapping. Latitude and time coordinates are retained. Cropping removes context pixels rather than blending the original RGB edges.

The native tiled decoder sees a wider canvas, so tile boundaries and overlap positions can change across the image. This can change interior detail as well as the seam. 128 and 384 remain selectable; neither is declared universally best.

## Adapter contract

`decode_circular` takes the **raw native H3 VAE** exposing `decode_output_shape(shape)` and `decode(latent, **kwargs)`. It expects the video latent in that implementation's native normalization. It does not reinterpret ComfyUI generic-wrapper latents or jointly packed audio/video streams.

For larger videos, the tested harness allocated a CPU float32 `output_buffer` of the extended decoded shape and passed it to the native decoder. Allocate that buffer after calculating the extended width, not at the cropped width. The returned crop may share its buffer; make a contiguous copy only if a downstream encoder needs one. Caller owns inference mode, dtype and model memory placement.

The original harness and this extraction use the same horizontal ContextPlan mapping and crop. Geometry/contract tests establish indexing equivalence; the exact extracted callable has not itself been re-run on a GPU.

## Limits

Correct neighboring context cannot guarantee semantic agreement between already inconsistent objects. A lower pixel mismatch does not establish an invisible join, preserved identity, stable fine texture or correct poles. Another decoder or upscaler may introduce its own boundary artifacts. Moving-camera behavior is a separate generation characteristic; movement is not required unless requested.

## Sampling extension

The [final shifted prediction](final-prediction.md) is a separate experimental intervention immediately before the final Euler update. It complements this decoder; it does not run VAE decoding on every sampling step.
