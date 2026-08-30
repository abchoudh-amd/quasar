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
decays through phase mixing. Its first maximum falls a quarter period after the
perturbation, at `t = pi/(2 omega_r) = 1.11`, and successive maxima follow every
`pi/omega_r = 2.22`; the 2000-step run therefore resolves the first four peaks
of the damped sequence directly, with no window to choose. The integration test
requires those peaks to decrease and fits their logarithmic envelope against the
Maxwellian damping rate, with a tolerance set by the measured seed-to-seed
spread.

`n_particles` is set by signal-to-noise rather than by cost. Discrete-particle
noise in the `k = 1` mode falls as `1/sqrt(N)`; at the earlier 8192 particles it
was within a small factor of the excited mode, so the envelope reached the noise
floor after about one e-folding and the late peaks were the amplitude coming
back up rather than damping. 65536 particles with a `4e-3` perturbation — still
linear, `delta_n/n` of order one percent — keep the mode above the floor for all
four peaks.
