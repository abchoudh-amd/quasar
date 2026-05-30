# Aspirational PIC example decks (not yet runnable)

Nine `examples/*/` PIC decks describe canonical plasma-physics validation cases
but **do not load** under the current `quasar.pic.io` schema — they target
features the loader/solver does not implement yet. They are kept (not deleted)
because each ships a descriptive `README.md` documenting the intended physics,
and they serve as the spec for the features below. They are intentionally
**not** wired into `tests/python/test_examples.py` (which only covers decks that
actually run end-to-end).

| example            | blocker (why `io.load` rejects it today)                    |
|--------------------|-------------------------------------------------------------|
| two_stream         | `units: normalized` (loader requires `SI`)                  |
| filtered_two_stream| `units: normalized`                                         |
| landau_damping     | `units: normalized`                                         |
| weibel             | `units: normalized`                                         |
| em_wave_propagation| `units: normalized` + `fields.initial` (EM-wave seeding)    |
| beam_in_channel    | dict-form `boundary` (`field`/`particle` as side-mappings)  |
| pec_cavity         | dict-form `boundary`                                         |
| magnetized_plasma  | external evaluator other than `biot_savart`                 |
| coil_confinement   | no `species` block (field-only confinement deck)            |

## To promote one to a tested example
1. Implement the missing loader/solver feature (e.g. a `normalized` unit mode,
   `fields.initial` seeding, or dict-form `boundary` parsing with the `field`
   side-map already modeled in `BoundaryConfig`'s sibling work).
2. Make `quasar.pic.io.load` accept the deck and `validate()` pass.
3. Add an integration test entry in `tests/python/test_examples.py` that runs a
   short `--steps-override` smoke run and asserts the documented diagnostic
   (e.g. two-stream electrostatic-energy growth, Landau damping decay rate).
4. Bring the example `README.md` up to the `single_loop` template (Run command +
   output keys + analytic reference).
