# filtered_two_stream

The `two_stream` setup with a current-smoothing pipeline enabled (4 binomial
passes plus one compensated-binomial pass) and TSC shapes. Demonstrates the
deck-configured filter pipeline damping short-wavelength grid noise while the
instability still grows.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/filtered_two_stream/input.yaml
```

`units: normalized`. Output `out.npz` keys match `two_stream` (Ex snapshots +
per-species state).

## Reference signature

The longitudinal field energy `sum(Ex^2)` still grows (the filter does not
suppress the physical instability, only grid noise). The integration test asserts
`energy[-1] > 10 * energy[0]`.
