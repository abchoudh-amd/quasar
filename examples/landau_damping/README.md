# landau_damping

Warm Maxwellian electrons with a small longitudinal velocity perturbation and a
fixed uniform ion background. The uniform initial charge and zero electric field
satisfy Gauss's law; the perturbation excites a Langmuir response whose envelope
is collisionlessly Landau damped (rather than growing as in `two_stream`).
The fundamental has `k lambda_D = 0.5`; the Maxwellian dispersion relation gives
`omega/omega_p = 1.41566 - 0.15336 i`.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/landau_damping/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ex`  — Ex snapshots (cadence 8).
- `field_ex`           — final Ex.
- `species_electron_*` — final particle state.

## Reference signature

The single-mode `Ex` response grows from zero, then its oscillation envelope
decays through phase mixing. The 2000-step run resolves three pre-noise-floor
mode-amplitude peaks. The integration test requires those peaks to decrease and
fits their logarithmic envelope against the Maxwellian damping rate, with a
finite-grid/finite-particle tolerance.
