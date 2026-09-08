# Circular decoding with a final shifted prediction

A tested extension to the circular decoder: let H3 predict the wrap boundary as an interior region **at the final denoising step**, then accept that prediction only in a narrow strip along the boundary. Apply the existing circular decoder to the finished latent.

[Four-page report card](H3-final-prediction-report-card.pdf) · [Measurements and image provenance](report-card/evidence.json)

## Pipeline

1. Sample normally up to the input of the final denoising step. Compute the ordinary clean prediction.
2. Roll the in-progress **video latent** horizontally by half its width. Run one additional H3 prediction at the same sigma. Roll its video prediction back.
3. Use the alternate prediction only in the first and last latent columns, at full strength. Keep the ordinary prediction everywhere else, including audio.
4. Complete the original final Euler update. Apply horizontal wrapped-context VAE decoding and crop the added margins.

The shift applies to **all temporal positions in the clip**. It is not a shift of the last video frame, a perspective projection, or a circular VAE decode at every sampling step. The accumulating trajectory stays in its original orientation. Model weights, LoRA weights and positional-encoding parameters are unchanged.

The tested strip is one latent column per side, corresponding to 16 output pixels per side. The decoder reference uses 128 px of context per side (eight latent columns). These are different widths serving different purposes. A 50-step schedule needs 51 model evaluations; a 100-step schedule needs 101. The alternate audio prediction is discarded.

## Reference algorithm

This is interface-level pseudocode. The executable sampler is [`final_shift.py`](../final_shift.py); the [deployment guide](deployment.md) covers its complete T2V runtime. It is not a ComfyUI graph node. `predict_clean` uses the host's H3 conditioning and normalized joint audio/video state. The host remains responsible for latent scaling and final unpacking.

```python
# Execute only for the final scheduled interval. state.video is B,C,T,H,W.
canonical = predict_clean(state, sigma)
shift = state.video.shape[-1] // 2
turned = state.with_video(state.video.roll(shift, dims=-1))
alternate = predict_clean(turned, sigma)
alternate_video = alternate.video.roll(-shift, dims=-1)

clean_video = canonical.video.clone()
clean_video[..., 0] = alternate_video[..., 0]
clean_video[..., -1] = alternate_video[..., -1]
clean = canonical.with_video(clean_video)  # retain canonical audio
state = state + (state - clean) / sigma * (next_sigma - sigma)

# After host-specific video unpacking / VAE normalization:
pixels = decode_circular(vae, final_video_latent, context_pixels=128)
```

The tested sampler requires horizontal half-turn alignment to H3's latent patch grid: latent width divisible by four (output width divisible by 64). It uses the original simple Euler schedule, BF16 H3 and FP16 native VAE. It rejects spatial image/video guides; those require consistent transforms of their conditioning and are not supported by this experimental path.

## Matched results

All rows use reviewed 360 LoRA strength 1.0. Compare each treatment against **ordinary sampling with the same circular-128 decoder**, not against ordinary decoding.

| Scene | Output | Frames | Steps | Control boundary MAE | Final-shift MAE | Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Temple | 1536 x 672 | 243 | 50 | 4.0981 | 3.3319 | 18.70% |
| Forest | 1536 x 672 | 243 | 50 | 8.3747 | 7.6123 | 9.10% |
| Temple | 1024 x 448 | 124 | 100 | 5.3703 | 3.7191 | 30.75% |

Boundary MAE averages the absolute RGB difference between the first and last columns across height, channels and every frame, on the 0-255 scale. A 30% reduction in this diagnostic is **not** a 30% improvement in visual quality. Distinct scenes' absolute scores should not be used to rank their perceived seams.

### Timing at 100 steps

One narrow correction, same temple and saved trajectory, same circular-128 decoder. Step numbers below are one-based.

| Correction step | Sigma | Ordinary steps afterward | Boundary error reduction |
| --- | ---: | ---: | ---: |
| 100 (final) | 0.108108 | 0 | 30.75% |
| 99 | 0.196721 | 1 | 13.61% |
| 95 | 0.433735 | 5 | 3.99% |

Step 99 has the same sigma as the old 50-step schedule's final step. The final placement still wins here. Timing changes sigma, Euler update size and the number of subsequent predictions together: the results do not isolate which mechanism explains the difference.

## Verification and limits

- Canonical and modified final predictions reconstruct both Euler outcomes. The reconstructed canonical output matches the comparison control's latent hashes.
- Final video latents outside the accepted edge columns and the entire audio latent match exactly. Decoding can spread pixel changes farther than the edited latent strip.
- The 100-step unchanged replay matches both streams and every decoded RGB frame. An initial-load/reuse discrepancy was found and excluded from this comparison using repeated matching controls; its root cause remains open.
- Lossless-source matched views show a less conspicuous join with scene layout retained. Residual structural seams remain. Native forest foliage has some softness; lower boundary error alone is insufficient.
- The 100-step circular-128 variant reduces global horizontal-gradient RMS by 0.68%. Pixels outside 128 px on each side are unchanged in all six saved snapshots. That is limited preservation evidence, not a universal detail or motion claim.
- This is evidence from two scene families, not a large independent corpus. The 100-step result is currently at the smaller canvas only. Image inputs, refinement/upscaling, multi-window continuation and correct poles remain separate validation tasks.

The native temple's extra prediction took approximately 27 seconds on the measured H200 run. This is sampling overhead from one run, not an end-to-end speed guarantee.

## Repository status

The wheel now includes the circular decoder, the extracted final-shift sampler and an optional complete H3 T2V runtime. The report-card evidence was produced by the original experiment harness; the separate [deployment validation](release-status.md) tests the clean wheel. No new training, model weights or license change is included.

## Opposite-side check

The report card now includes yaw-0 views opposite the wrap. The middle half of the ERP is pixel-identical between circular-128 control and treatment in temple/forest frame 97 and the 100-step temple frames 0 and 123. These are selected-frame checks; image hashes and exact regions are in `report-card/evidence.json`. The shifted prediction's new boundary is discarded with the rest of its interior. The deployment runtime additionally checks the opposite side across every decoded frame.
