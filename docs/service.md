# Spherical H3 endpoint

The deliverable is one configurable fal endpoint backed by this repository. It
combines the tested longitude correction with optional video repair at either
pole. It also accepts an existing complete ERP video for pole repair alone.

## Processing

```text
Generation prompt
  → native H3 sampling + optional final shifted prediction
  → ordinary or circular VAE decode
  → unrepaired ERP comparison
  → optional top and/or bottom perspective-video repair
  → backproject repair deltas onto the ERP
  → complete ERP video + exact-settings receipt
```

Existing-video input starts at the unrepaired ERP. It does not pass through the
native sampler or circular decoder again. The response explicitly records which
stages ran. A circular decoder needs latent/model access; this endpoint cannot
retroactively apply that decoder to another provider's encoded MP4.

Each selected pole is projected to a 512×512, 90-degree perspective video. A
small angular mask has a 16-degree core and a feather ending at 22 degrees. The
sampling mask covers complete 32-pixel model cells; the final pixel composite
uses the original soft mask. No geometric unpinching is applied.

The repair stage receives the full projected video through the VAE and a
**separate text instruction**. Its default is exactly:

> Repair the polar distortion.

There is no automatic prompt expansion or scene description. The text encoder
receives no images at this stage. Top and bottom instructions are independently
editable, including an empty string. Empty text was less reliable in our tests
and sometimes introduced unrelated objects; it does not remove the model's
conditioning architecture.

Repair uses BF16 H3, FP16 VAE, no 360 LoRA, a fixed zero audio latent, and
`res_multistep` with an explicit linear sigma schedule from 1 to 0. The default
is 32 steps. Original ERP pixels outside the projected repair delta are retained
before delivery compression. Both poles are extracted from the same original
video, so one repair cannot alter the other's input.

## Inputs

| Field | Default | Meaning |
|---|---|---|
| `prompt` | empty | Main panorama prompt, required unless `video_url` is supplied |
| `video_url` | absent | Existing complete ERP video; leave `prompt` empty |
| `width`, `height` | 1536, 672 | Also supports the explicit diagnostic size 1024×448 |
| `frames` | 124 | Current service boundary; incompatible lengths are rejected |
| `steps` | 50 | Native generation steps, range 1–100 |
| `seed` | 2026090919 | Native sampling seed |
| `circular_decode` | true | Wrapped VAE context on newly generated latents |
| `context_pixels` | 128 | 128 or 384 pixels of horizontal context |
| `final_shift` | true | One extra shifted prediction at the final diffusion step |
| `repair_top`, `repair_bottom` | true, true | Independent pole-repair switches |
| `top_prompt`, `bottom_prompt` | repair instruction above | Never derived from the main prompt |
| `repair_steps` | 32 | Perspective repair steps, range 1–40 |
| `repair_seed` | 2026090911 | Same declared seed for each pole |

Native generation uses the reviewed ERP LoRA at strength 1.0 and the original
`simple` Euler schedule. Final shift applies to all video frames at the final
diffusion step, accepting only one latent column at each edge. It is not a
last-video-frame edit.

All output is 24 fps and **silent**. Audio synthesis/preservation, arbitrary
durations, image-conditioned native sampling and refinement/upscaling are not
implemented in this service version. No unsupported input is silently shortened,
resized, or given fewer steps. Non-default repair settings are exploratory, not
covered by the default-recipe evidence.

## Example configurations

Keep the scene description in `prompt`. Leave both pole instructions at their
default unless deliberately testing a different repair direction.

```json
{
  "prompt": "equi360, an ancient temple under a golden yellow sky",
  "repair_top": true,
  "repair_bottom": true,
  "top_prompt": "Repair the polar distortion.",
  "bottom_prompt": "Repair the polar distortion."
}
```

Set both repair switches to `false` to generate with longitude corrections alone.
Set `circular_decode` and `final_shift` to `false` as well for the native baseline.
To repair an existing clip, supply `video_url`, leave `prompt` empty, and choose
one or both poles. The result receipt records that native sampling and circular
decoding were skipped. This is one endpoint with independent controls, rather
than separate top/bottom deployments that each load their own models.

## Outputs and comparison

- `video`: final complete ERP, with selected repairs.
- `unrepaired_video`: complete ERP immediately before pole repair.
- `canonical_video`: for generation requests, the ordinary final prediction
  decoded with the same context. Compare it to the unrepaired video to isolate
  the final shifted prediction.
- `receipt`: exact request, stage settings, model/source hashes and output
  hashes. This distinguishes technical completion from visual assessment.

Lossless masters and latent receipts are kept under the request ID on fal
persistent storage. MP4s are review/delivery copies; compression can change
pixels outside the repair. CDN URLs follow the account's retention policy.

## Build and deploy

Install the fal CLI and `uv` separately and configure your fal account. The
bundle builder uses `uv build`. From this checkout:

```bash
pip install '.[service,torch]'
python -m unittest discover -s tests -v
python scripts/build_service.py /tmp/h3-spherical-service
fal run /tmp/h3-spherical-service/service_app.py::H3SphericalService --auth private
```

Use the printed temporary endpoint to validate the built artifact before a
persistent deployment. The service uses a SHA-pinned Comfy source archive and
model manifest. The wheel contains all pipeline code; it imports nothing from
the research workspace. No weights, data, keys or recordings are bundled.

After validation:

```bash
fal deploy /tmp/h3-spherical-service/service_app.py::H3SphericalService \
  --auth private --strategy recreate --reset-scale
```

The default capacity is one H200 worker, no warm minimum, no speculative buffer,
and a 30-second idle keepalive. `recreate` avoids proactively starting a worker
during deployment. Requests have a 55-minute application deadline and subprocess
cancellation. This is a request time limit, not a whole-account spending cap.

Use fal's queue client so closing a browser does not interrupt inference:

```python
import fal_client

handle = fal_client.submit("YOUR_ACCOUNT/h3-spherical", arguments={
    "prompt": "equi360, an ancient temple under golden skies, stationary camera",
    "repair_top": True,
    "repair_bottom": True,
})
print(handle.request_id)  # Save this before polling. Do not resubmit on timeout.
result = handle.get()
print(result["video"]["url"])
```

Each request has its own durable claim and directories. A completed duplicate
returns the saved result; an interrupted claim fails rather than silently
repeating paid inference. Cancellation kills the current GPU subprocess.
Generation and repair run in separate processes to release encoder/model RAM
and prevent the panorama LoRA leaking into perspective repair. This favors
reproducibility over warm-model throughput; memory residency is a later
optimization.

## Sharing with a reviewer

Private fal endpoints accept only the owner/team's keys. Shared mode accepts
other fal users' keys and bills those users, but requires fal admin enablement.
Public mode accepts unauthenticated requests and bills the owner. Do not publish
an account key to make a private endpoint accessible. The intended reviewer
handoff is shared mode once enabled, or an explicitly agreed restricted access
arrangement.

Sources: [deployment and authentication](https://fal.ai/docs/documentation/deployment/deploy-to-production),
[scaling](https://fal.ai/docs/documentation/deployment/scale-your-application),
[queue](https://fal.ai/docs/documentation/model-apis/inference/queue),
[cancellation](https://fal.ai/docs/documentation/development/handle-cancellations).

## Evidence and extension boundary

The prior six-view prompt ablation supports the short instruction as the most
consistent tested choice at one fixed seed. It does not prove universally
correct poles. Service integration validation is recorded separately in the
release status; packaging alone is not a new visual result.

`service_schema.py` is the interface; `service_worker.py` selects generation or
repair; `h3_runtime.py` owns native generation; `polar_pipeline.py` owns projection,
conditioning and composition; `polar_kernel.py` owns masked H3 sampling. The fal
wrapper handles requests, files and process lifetime. This separation permits
new model access or a different repair backend without changing the public
top/bottom controls.

The backend is the publicly available Comfy-converted H3 FL2VA weights. It is
not fal's internal Max or Turbo endpoint, and does not imply access to those
implementations. Reusing this design there requires the relevant model/runtime
access from fal.


The projection executable currently comes from the worker's system FFmpeg,
with the pinned `imageio-ffmpeg` package as a fallback. Projection is therefore
not guaranteed bit-identical across different base images. Validation uses the
actual packaged service's projected inputs; it is an integration check rather
than an exact replay of older repair videos.
