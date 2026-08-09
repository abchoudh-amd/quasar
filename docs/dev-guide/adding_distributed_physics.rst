.. meta::
   :description: Extend Quasar distributed execution with another physics slice.
   :keywords: Quasar, distributed runtime, MPI, multi-GPU, developer guide

Adding a physics slice to distributed execution
================================================

The distributed layer is a cross-cutting execution axis, not a third copy of a
physics solver. A distributed physics slice partitions one canonical global
state into tiles, drives the existing solver phases on endpoint workers, and
reassembles only the state required for diagnostics or restart.

Use the following checklist when adding a physics package beyond PIC or MHD.

#. **Define the native tile runtime.** Add a public runtime interface under
   ``include/quasar/distributed/`` and its implementation under
   ``src/distributed/``. Keep MPI and parallel-HDF5 ownership in
   ``MpiRuntime`` / ``Transport`` / the checkpoint layer. Solver internals must
   be reached through a named phase/access interface rather than new blanket
   friendship.

#. **Define canonical and owned state.** Specify one topology-independent global
   seed/restart representation and one rank-local owned-shard representation.
   Record stagger, offsets, and extents explicitly. Validate complete,
   non-overlapping coverage before mutating solver state.

#. **Derive the topology reach from the selected scheme.** The high-level
   runner must pass the reconstruction/stencil halo to
   ``RuntimeSession.select_topology(global_nx, global_ny, decomposition,
   minimum_tile_width)``. The low-level binding deliberately has no default for
   either ``decomposition`` or ``minimum_tile_width``. A rejected decomposition
   reports both the requested shape and required reach.

#. **Make boundary capabilities explicit.** A registered field/fluid boundary
   used by a tile solver must advertise its ``ghost_continuation_mode()``; do
   not branch on its registry name in either the solver or distributed runtime.
   Coordinator-owned tile interfaces use the reserved ``"internal"`` closure
   and must also report ``is_internal_cut()`` so local physical ghost fills and
   particle closures do not overwrite exchanged state.  Keep this name out of
   user deck vocabulary.

#. **Keep collective control flow aligned.** Rank-local parsing, allocation, and
   filesystem work must be followed by ``collective_require`` before the next
   MPI/HDF5 operation. All ranks enter native phases in the same order, even
   when a rank owns no endpoint for that phase. Fatal post-mutation failures
   poison the runtime; collective ``close`` remains available.

#. **Add checkpoint compatibility and continuation state.** Physics state goes
   into typed parallel-HDF5 datasets. Python diagnostic history uses the bounded
   non-pickle fragment codec in ``quasar._checkpoint_diagnostics``. Restart must
   work across a changed rank count, endpoint count, and decomposition.

#. **Register the Python orchestrator from the physics package.** The package
   ``__init__.py`` calls ``quasar.distributed._register_runner`` with a lazy
   runner callable. The physics-neutral ``quasar.distributed`` module must not
   enumerate physics names. Reuse ``quasar._distributed_helpers`` for collective
   local phases, policy signatures, and atomic NPZ/JSON publication.

#. **Expose and bind the complete surface.** Extend ``RuntimeSession`` with the
   physics operations and a ``<physics>_runtime_available`` probe. Add the pure
   Python runner to the staged-source list in
   ``bindings/python/CMakeLists.txt``. Do not make availability mean only that
   MPI/HDF5 compiled: it means the end-to-end runner is usable.

#. **Test all placement and continuation paths.** Add CPU orchestration tests
   with a stateful fake session, native single-/multi-GPU and multi-rank tests,
   gathered and sharded diagnostics, checkpoint cadence, corruption rejection,
   and restart with a different decomposition. GPU tests need
   ``RESOURCE_GROUPS`` and the common CTest launcher. Fault-injection tests are
   enabled only by the default-off ``QUASAR_DISTRIBUTED_TEST_HOOKS`` option.

#. **Document and demonstrate the launcher.** Add a self-contained deck,
   scheduler launcher, README, and ``test_examples.py`` contract test. Update
   :doc:`../user-guide/distributed_simulation` and ``CHANGELOG.md`` when the
   supported physics list or runtime policy changes.

Build and test the slice with the matching ``*-distributed-debug`` preset
first, then the release and ``-all`` variants. Distributed test presets load
``tests/ctest_gpus.json``; sites with a different visible GPU inventory should
pass their own CTest resource specification.
