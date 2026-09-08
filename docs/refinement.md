# Focused refinement and upscaling validation

The official H3-Regenerate-2K route feeds the low-resolution video and original context back into H3 to regenerate higher-resolution output. MiniMax currently provides it as a hosted module. Our accessible native refinement path is a separate experiment, not a reproduction of that closed module. [Official description](https://github.com/MiniMax-AI/MiniMax-H3#h3-regenerate-2k).

A community ComfyUI implementation also explores low-to-high-resolution denoising handoffs and trajectory-guided refinement. Its author explicitly distinguishes it from the official module. We have not run or validated it. [Flow-Aligned-Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate).

## Next controlled test

Use the saved lossless circular128 temple video as a common input, retaining all243 frames at24fps. Upscale the complete ERP canvas from1536×672 to2048×896 with identical Lanczos resampling. Keep a resized-only control.

1. Encode the common higher-resolution input once with the ordinary native VAE.
2. Save a no-refinement decode control to measure reconstruction loss.
3. Run one50-evaluation native refinement pass, denoise0.2, using the original prompt, reviewed LoRA1.0 and the same frame count. Preserve the source audio latent.
4. Save the refined latent once; decode it natively, with128px context, and with384px context.
5. Compare the same seam and interior regions, camera, frame and angular FOV. Inspect at source and target display scale so extra pixels are not mistaken for recovered detail.

This first pass isolates whether circular **decoding** remains useful after refinement. A later matched ordinary/circular **encoding** pair can test the input side without confounding this first result. Fixed pixel margins and fixed angular margins are also different variables: at2048px,128px covers a smaller longitude angle than at1536px. Record that distinction before tuning margin width.

Judge carved stone detail, edge halos, texture smearing, semantic continuity, temporal stability and poles separately. Include lossless stills and complete ERP videos. Boundary MAE alone cannot choose a winner.

Status: prepared protocol, not yet run. The larger generation corpus remains deferred while this focused question is tested.
