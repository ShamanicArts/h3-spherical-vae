# Static comparison viewer

Serve this directory over HTTP. Video media is excluded from Git and the source release. A bare clone does not provide playable comparisons; use the README PNGs and evidence JSON until a separate media bundle is supplied. The page needs no account, model, backend generation service or analytics. Use a static host with byte-range support for efficient seeking. The existing selected-pair Blob fallback supports simple development servers.

The renderer uses one shared offscreen WebGL context, and only the selected A/B videos decode. Free look, pitch, horizontal field of view and video frame are synchronized. Native CRF16 delivery is the default; full-resolution CRF20 review copies have shorter GOPs for seeking. Selected lossless PNGs support still-frame detail checks. These quality modes do not change generation settings.

The mono sphere mapping is retained from the author's earlier 360 viewer. Renderer source is included in `sphere-viewer.ts`; Three.js is bundled locally with its license. Rebuild from the repository root with `npm ci --ignore-scripts && npm run build:viewer`. The comparison UI is self-contained in `index.html`.

This demo contains two ordinary-sampling examples, each decoded native/128/384 from one saved latent. It does not demonstrate a custom sampler, polar repair, an upscaler, or inference in the browser.
