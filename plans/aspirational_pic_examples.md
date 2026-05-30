# PIC example decks — promoted to runnable + tested

All nine PIC example decks under `examples/*/` now load through `quasar.pic.io`,
run end-to-end via `python -m quasar.pic.cli run`, and have an integration-test
entry in `tests/python/test_examples.py::PicAspirationalExampleTests` asserting
their characteristic signature. Each ships a `README.md` in the `single_loop`
template (run command + output keys + reference signature).

The features that previously blocked them have all landed:

| former blocker                         | resolution |
|----------------------------------------|------------|
| `units: normalized` rejected           | CLI applies the plasma `Normalization`; `normalized` is an identity pass-through |
| `fields.initial` seeding unimplemented | `fields.initial` schema + `EmPic2D3V.seed_field` binding (`seed_perturbation`, `seed_em_wave`) |
| dict-form `boundary` (field/particle)  | `_parse_side_map` handles scalar / list / side-keyed dict; `set_field_side` binding |
| field BCs inert                         | ghost-aware adjoint Yee stencil + per-step ghost fill (see `field_bc_heisenbug.md`) |
| evaluator other than `biot_savart`      | `IFieldEvaluator` registry bound; `uniform`/`dipole`/`gradient` selectable |
| species-less (field-only) deck          | a deck may define species, an external field, OR `fields.initial` |

Signatures asserted by the tests: two-stream / filtered-two-stream grow the
longitudinal field energy; weibel grows transverse `Bz`; em-wave and pec-cavity
keep EM energy bounded; beam-in-channel confines all particles; landau-damping,
magnetized-plasma, and coil-confinement load/seed/step to finite output with the
expected external field present.
