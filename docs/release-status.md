# Release status

Experimental source release, version 0.2.0. An original-code license has not yet been selected.

## Included

- Circular wrapped-context decoder and geometry utilities.
- Executable final-step shifted prediction sampler, with narrow acceptance and unchanged audio.
- Full T2V runtime: prompt encoding, pinned H3/LoRA loading, sampling, matched decodes, video exports and audit receipts.
- Buildable fal deployment bundle from the wheel, with SHA-pinned upstream source and weights.
- Five-page report card, including opposite-longitude views.
- Separate polar coordinate module, analytic fixtures and controlled research plan.

## Verified locally

The geometry, decoder, sampler and polar-coordinate tests pass (25 tests). The wheel builds and imports outside the checkout; its sampler tests pass in isolation. The extracted sampler matches the private research implementation bit-for-bit in a deterministic 100-step contract comparison. These CPU checks do not substitute for a real model deployment.

## Real-model validation

A fresh fal H200 validation completed from the built wheel and fresh text conditioning: **1024×448, 124 frames, 50 steps plus one final evaluation**, H3/text BF16, LoRA 1.0, native VAE FP16. All six control/treatment decodes completed.

| Decode context | Ordinary boundary MAE | Final-shift MAE | Reduction |
|---|---:|---:|---:|
| 0 px | 11.1098 | 7.9060 | 28.84% |
| 128 px | 4.7707 | 3.3310 | 30.18% |
| 384 px | 4.6132 | 3.5638 | 22.75% |

The middle half of the ERP is pixel-identical across all 124 frames in all three comparisons. Exterior latent columns and audio are exact. Both final Euler updates reconstruct exactly locally; all **744 lossless decoded frames** match the recorded hashes. Selected seam views show a reduced line with residual texture mismatch.

Sampling took **138.84 seconds**. Runner lifetime, including setup, hashing and collection, was **581.72 seconds**; estimated compute **$0.73** at $4.50/hour (not an invoice). Compute termination was verified. This was one authorized generation, with controls captured in the same final forward call.

The final source includes one reporting-only follow-up: mutable progress documents are excluded from the durable artifact manifest. That fix passes a regression test and rebuilt-wheel tests; it does not change inference. GPU-tested and released source/wheel hashes are distinguished in [the validation receipt](deployment-validation.json). No second GPU generation was needed for that reporting fix.

## Limits

This is a T2V runtime, not a packaged ComfyUI graph node. It saves audio latents but exports silent videos. Image-conditioned sampling, upscaling/refinement, long continuations and polar repair are not established by this release. Core GPU dependency versions are specified; the transitive environment is not a fully locked dependency set. Model and upstream implementation licenses remain separate.
