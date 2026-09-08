# Seam-local blur and noise: proposed experiments

Status: hypothesis and test design, not an implemented or validated feature of v0.1.

The circular decoder supplies neighboring context; it cannot guarantee that an already inconsistent latent describes one coherent object. The next question is whether relaxing a narrow band around that join helps the model resolve conflicting structure.

## Priority: blur during generation

The intended next intervention is step-dependent blur during sampling, not an additional repair pass. Regularize the predicted clean video latent in a narrow periodic seam band while structure is forming; taper the strength to zero before the final detail-forming steps. This is a hypothesis, not an established result.

A candidate operation is `clean <- clean + strength(step) * mask * (wrap_blur(clean) - clean)`. The mask spans both ERP edges as one region and fades to zero at its inner boundaries. Blur samples across the wrap. Start with horizontal spatial blur only; keep latitude and temporal filtering out of the first comparison.

Apply this to a clean prediction derived using the actual H3 flow scheduler, then convert back consistently before the solver update. Do not indiscriminately blur the noisy state: that also changes the noise distribution. H3 model parameters are shared across positions and are not an addressable seam band; step-dependent activation/prediction blending is the more direct intervention.

Compare the existing circular decoder alone against early-only and early-to-middle prediction blur with identical initial noise and total model evaluations. Preserve the original unmodified prediction as the control. Use tapered strengths that end before the final steps, rather than assuming the first pure-noise steps contain a meaningful seam. Sweep strength and band width separately. Inspect whether later steps restore detail coherently, restore the old seam, or create new transitions at the mask edges. Do not claim the seam will disappear before this is tested.

## Other controls and alternatives

- **RGB blur after decoding:** a cheap control. Smooth only a tapered band straddling the wrap with circular indexing. It may reduce edge MAE while losing detail; it does not ask the model to reconstruct geometry.
- **Latent blur before decoding:** low-pass the clean latent inside the same band. This tests whether the discontinuity is carried by high-frequency latent structure. Latent channels are learned features, so smoothing can change identity or geometry rather than simply soften pixels. The VAE alone is not a diffusion denoiser.
- **Noise plus a denoising pass:** perturb the band at a defined scheduler level and let the model reconstruct it with context from both sides. Keep the rest of the canvas on a consistent reference trajectory. This can revise geometry but costs additional model evaluations and may change identity or introduce flicker.

Blurring the injected noise is a fourth, distinct variable: it changes noise correlations away from the normal sampling distribution. Do not assume blurred noise behaves like blurred image detail.

[SDEdit](https://arxiv.org/abs/2108.01073) establishes noise-then-denoise editing; [RePaint](https://arxiv.org/abs/2201.09865) demonstrates masked reconstruction using a diffusion prior. These motivate the experiment. Neither paper establishes that the same construction works on H3's video flow model without adapting its scheduler and conditioning.

## Controlled first test

Use the saved temple latent and its existing circular-128 decode as the baseline. Retain the seed, full ERP dimensions, duration and common circular decoder across arms. Compare no intervention, tapered RGB blur, tapered latent blur, and one explicitly budgeted masked denoising arm. Vary band width and strength independently rather than changing them together.

Treat the left/right band as one periodic region. If a seam-centred chart is used, transform every spatial conditioning/reference consistently and restore the original longitude afterward; earlier chart changes introduced distortions. Keep the unedited region at the current scheduler's noise level during sampling, not as clean pixels spliced into a noisy latent. Do not invent a DDPM noise formula for H3's flow schedule.

Sample one coherent spatiotemporal noise tensor according to the model's expected convention and reuse it across matched arms. Do not generate a different noise pattern at each edit iteration. Temporal stability must still be inspected.

Judge the actual join, carving identity, local detail and temporal behavior alongside edge MAE. A lower error achieved by smoothing away the carving is not the desired improvement. This test is separate from the higher-resolution refinement protocol and requires its own run receipt before paid execution.
