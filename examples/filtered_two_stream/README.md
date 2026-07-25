# filtered_two_stream: longitudinal-filter invariance

The deterministically perturbed `two_stream` setup with a current-smoothing
pipeline enabled (4 binomial passes plus one compensated-binomial pass).  Every
other physical and numerical input, including the CIC shape, matches
`two_stream`. A fixed background neutralizes the equilibrium electron charge.

This is deliberately an **invariance** example, not a smoothing demonstration.
The longitudinal instability deposits `Jx`; Quasar's charge-continuity-safe 2D
filter smooths only the transverse `Jz` component and leaves `Jx`/`Jy` bitwise
unchanged within each filter operation. Independent GPU runs can still differ
at floating-point roundoff because current deposition uses atomic sums whose
addition order is not deterministic. Consequently the configured pipeline must
not change this deck's physical evolution beyond that roundoff. Transverse
filter response and high-wave-number attenuation are covered by the filter
impulse-response and through-solver tests.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/filtered_two_stream/input.yaml
```

`units: normalized`. Output `out.npz` keys match `two_stream` (Ex snapshots +
per-species state).

## Reference signature

The integration test fits the longitudinal `k=1` Fourier amplitude over the same
linear window as `two_stream` and checks its rate against the cold two-beam
dispersion relation. The test verifies that enabling a transverse-current
filter cannot perturb the charge-carrying longitudinal mode.
It also compares a short filtered trajectory directly with the matched
unfiltered deck; the saved longitudinal fields and particle state must agree to
a scale-aware floating-point roundoff tolerance.
