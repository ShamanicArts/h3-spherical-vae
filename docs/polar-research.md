# Polar projection: next experiments

The seam baseline stays fixed: ordinary H3 sampling, one local final shifted prediction, circular-128 decoding. Polar work is a separate experimental path, not a new default.

## What we already learned

Horizontal context helps the decoder see the opposite side. Extending that idea over the poles has not yet produced convincing unpinching. In nine saved-latent decodes (three scenes, three context treatments), reflected rows with a half-longitude turn changed colour/detail but did not reliably improve polar geometry. Correct pixel coordinates do not establish that learned VAE channels transform the same way.

Direct perspective remapping of H3 latent channels produced ribbed stone before the repair model ran. Projecting decoded pixels and re-encoding through the native VAE avoided that particular failure. Its one-step seam repair was still weaker than shifted ERP prediction. That is an implementation lesson for a pole-facing view, not evidence that it will repair poles.

An ERP image stretches near its poles by construction. We must judge content in pole-facing perspective views, distinguish this valid projection from pinched structures, and use a known correct spherical reference where possible. Uniform rows alone are not a useful success criterion.

## First surface: measure the coordinate and codec losses

Use analytic spherical textures, lines and objects with known geometry, plus curated real panoramas. Render each reference directly into north/south pole views. Compare:

1. Pixel projection and inverse projection only. This establishes the interpolation floor.
2. Ordinary VAE encode/decode, no model prediction.
3. Circular encode with the decoder unchanged.
4. Circular decode with the encoder unchanged.
5. Circular encode and decode together.
6. Pixel rotation putting a pole at the equator, then native VAE encode/decode and inverse pixel rotation.

Keep pixel dimensions, VAE tiling and source frames recorded. Compare to the exact same resampling-only reference, rather than attributing every round-trip change to the VAE. The longitude-only half-turn is an exact permutation; pole-to-equator rotation is a resampling operation and requires this extra control.

Record pole-view reconstruction error, edge displacement/line straightness against the known view, retained high-frequency detail, temporal consistency, and changes outside the polar cap. Use solid-angle weighting for whole-sphere errors so ERP's heavily oversampled poles do not dominate the score. Present full ERP, north/south views and an equatorial control at fixed FOV.

## Second surface: local model repair

Only after those controls work, test a short late denoising pass on a pole-facing pixel/VAE chart. Start with identity/no-prediction and an ordinary-model control. Transfer the model-induced difference relative to the unchanged codec round trip; constrain the accepted difference to a smoothly tapered polar cap. Test both poles independently and inspect the cap boundary and all other directions. Compare one prediction with a short late schedule. No direct latent interpolation is presumed safe.

Use a deliberately pinched synthetic source with a correct reference to test recovery, then move to generated/real scenes. This distinguishes reducing a metric by smoothing from reconstructing useful structure. A paired correction LoRA is a later training hypothesis; it needs correct targets and diversity before training makes sense.

## Architecture research after the controlled projection test

[SphereDiff](https://arxiv.org/abs/2504.14396) and its [official implementation](https://github.com/pmh9960/SphereDiff) use a spherical latent representation with pretrained diffusion models. It motivates coordinating multiple views, but does not establish that our packed H3 channels can be bilinearly projected as pixels.

[SpheRoPE](https://arxiv.org/abs/2606.32033) and its [official implementation](https://github.com/orhir/SpheRoPE) replace positional encoding with spherical/periodic structure. Its published examples include Flux and LTX-Video; transferring it to H3 is a new experiment. Our earlier broad positional changes sometimes altered composition. Keep those experiments separate from the local repair controls, with matched seeds, unchanged prompts and explicit detail/motion checks.

## Promotion rule

A candidate must improve the pole-facing view across more than one scene, retain detail/motion, and preserve the established seam gain. Report failures as well as the best view. Never promote a lower edge-pixel error on its own.

Status: plan and coordinate fixtures only. No polar H3 generation is claimed by this document.

The initial CPU fixture is reproducible with `PYTHONPATH=. python scripts/polar_fixture.py <new-output-directory>` (NumPy and Pillow). On a 1536×672 analytic ERP, its 512×512 north/south 75-degree views have float RGB RMSE 0.00001193 versus direct evaluation of the spherical signal. This measures interpolation on a smooth known signal; it is not a VAE or generation result.
