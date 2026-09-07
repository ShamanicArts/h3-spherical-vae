# Spherical VAE context experiments

Small, backend-independent coordinate utilities for controlled ERP boundary
experiments. This repository has a fresh history and contains selected reusable
code only. It contains no models, personal experiment logs, media, datasets,
cloud credentials or deployment controllers. It has not been published.

The first component constructs horizontal circular context and experimental
pole-crossing context for a pixel-centred ERP grid. It returns explicit source
indices and crop bounds. A NumPy reference and optional PyTorch application
share the same coordinate plan.

Pole-crossing context reflects latitude and shifts longitude by 180 degrees.
That is a coordinate relationship on a sphere, **not a claim that learned VAE
channels are equivariant under the transformation**. Actual H3 quality needs a
matched model experiment. No pole-aware H3 decode has been validated here yet.

Run local checks with Python and NumPy:

```sh
python -m unittest discover -s tests -v
```

The checks cover identity cropping, immutable inputs, native latent dimensions,
periodic corners, north/south independence and agreement with analytic unit
sphere coordinates across each pole. The optional PyTorch adapter requires
PyTorch supplied by the model environment.

This is original utility code. Upstream model implementations and weights are
not bundled; their licenses remain separate. A project distribution license
has not yet been selected.
