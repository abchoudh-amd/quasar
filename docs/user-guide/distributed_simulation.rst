.. meta::
   :description: Run one Quasar PIC or MHD simulation across MPI processes and GPUs.
   :keywords: Quasar, MPI, multi-GPU, distributed PIC, distributed MHD, checkpoint

Run distributed PIC and MHD simulations
=======================================

Quasar can divide one PIC or MHD mesh into a two-dimensional grid of GPU
tiles.  MPI processes may span nodes, and each process owns the same positive
number of GPUs.  The MPI launcher chooses the process placement; Quasar does
not store placement in the physics deck and does not provide a ``--ranks``
option.

The existing serial path is unchanged.  A plain
``python -m quasar.pic.cli run`` or ``python -m quasar.mhd.cli run`` command
still uses one process and one GPU.  Supplying a distributed option selects the
tile runtime, including when MPI world size is one.

Build distributed support
-------------------------

``QUASAR_ENABLE_DISTRIBUTED`` is a tri-state CMake cache setting:

``AUTO``
   The default.  Enable the complete feature only when CMake finds MPI C++
   3.1 or newer, Threads, and parallel HDF5 C 1.10 or newer, and a combined
   ``MPI_Init_thread`` / ``H5Pset_fapl_mpio`` link probe succeeds.

``ON``
   Require those dependencies.  Configuration stops with one diagnostic that
   lists every missing or incompatible dependency.

``OFF``
   Skip MPI and HDF5 discovery and build only the serial runtime.

For example, configure a gfx950 build with required distributed support:

.. code-block:: shell

   cmake -S . -B build/hip-gfx950-distributed-release \
     -DQUASAR_ENABLE_DISTRIBUTED=ON \
     -DCMAKE_HIP_ARCHITECTURES=gfx950 \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build/hip-gfx950-distributed-release --parallel

The ``hip-gfx942-distributed-release`` and
``hip-gfx950-distributed-release`` configure, build, and test presets provide
the equivalent project defaults. Matching ``-distributed-debug`` configure and
build presets enable the dedicated fault-injection hooks, while ``-all`` test
presets include the slow example suite. ``QUASAR_DISTRIBUTED_TEST_HOOKS`` is
default-OFF and should not be enabled in production binaries.

``quasar.distributed`` is importable in every build.  Use
``quasar.distributed.is_available()`` to query complete runtime availability;
an explicit distributed request in a serial-only build raises
``DistributedUnavailableError`` rather than silently falling back.

Launch a run
------------

The following command launches two ranks on one node.  Each rank receives two
visible GPUs, giving four virtual endpoints arranged as a ``2x2`` tile grid:

.. code-block:: shell

   export PYTHONPATH="$PWD/build/hip-gfx950-distributed-release/python"
   mpiexec -n 2 python -m quasar.mhd.cli run \
     examples/orszag_tang/input.yaml \
     --devices auto \
     --decomposition 2x2 \
     --transport staged

Replace ``quasar.mhd`` with ``quasar.pic`` and provide a PIC deck to run PIC.
One rank with multiple visible GPUs uses the same runtime and needs no MPI
launcher, for example ``--devices 0,1 --decomposition 2x1``.

``--devices auto`` uses the GPU visibility established by the launcher.  On a
node where ranks share one visibility mask, Quasar partitions that node-local
pool between the ranks.  It also accepts masks that the launcher has already
made disjoint.  An explicit list such as ``--devices 0,2`` is an eligible
node-local pool, not a claim that every rank owns both devices.  Startup
compares physical UUID or PCI identities on each node and collectively rejects
overlapping ownership, duplicate devices, a zero-GPU rank, or unequal GPU
counts per rank.

``--decomposition auto`` chooses a factorization of the total endpoint count
that balances tile aspect ratio and halo surface.  An explicit ``PXxPY`` must
have ``PX * PY`` equal to the total GPU count.  Remainder cells are assigned to
lower tile coordinates.  Quasar rejects a tile that is too thin for the
selected PIC or MHD stencil.

The low-level native ``RuntimeSession.select_topology`` API requires both the
decomposition policy and ``minimum_tile_width`` explicitly. High-level PIC and
MHD runners derive that width from the selected field/reconstruction stencil;
custom native callers must do the same. A rejected explicit shape identifies
the requested decomposition and required stencil reach.

Intra-process neighbours use peer copies when the devices support them and
otherwise use pinned-host staging.  Inter-process ``--transport staged`` uses
pinned host buffers and nonblocking MPI.  ``--transport auto`` selects direct
device-buffer MPI only when the MPI implementation reports ROCm-aware support
and a collective startup probe succeeds; otherwise it resolves to ``staged``.
An explicit ``--transport direct`` request fails collectively when either
capability check fails instead of silently falling back.  Run telemetry records
the requested and resolved policies, byte counts for each path, and the direct
capability checks.

All MPI and parallel-HDF5 calls run on the orchestration thread under
``MPI_THREAD_FUNNELED``.  Quasar reuses a sufficiently initialized external MPI
runtime and finalizes MPI only when Quasar initialized it.

Select diagnostics
------------------

``--diagnostics-layout gathered`` is the default.  Rank zero writes the same
NPZ path, keys, shapes, and PIC ``--write-every`` snapshot names as a serial
run, so existing post-processing continues to work.

``--diagnostics-layout sharded`` writes one owned-data NPZ file per GPU
endpoint and publishes ``<output-stem>.manifest.json`` last.  The manifest uses
schema ``quasar-diagnostics-shards/v1`` and records the physics kind, global
shape, step and time, decomposition, rank/device/tile mapping, offsets, owned
extents, and shard paths.  Its presence is the completion marker.  Validate and
inspect it from Python with:

.. code-block:: python

   from quasar.distributed import read_diagnostics_manifest

   manifest = read_diagnostics_manifest("out.manifest.json")
   print(manifest.physics, manifest.global_shape, manifest.decomposition)

Sharded NPZ files and gathered NPZ snapshots are diagnostics only; neither is
accepted by ``--restart``.

Checkpoint and restart
----------------------

Use an explicit parallel-HDF5 path for restartable state:

.. code-block:: shell

   mpiexec -n 2 python -m quasar.pic.cli run examples/two_stream/input.yaml \
     --devices auto --decomposition 2x1 \
     --checkpoint /scratch/$USER/two-stream.h5 \
     --checkpoint-every 100

``--checkpoint PATH`` writes a final committed checkpoint.
``--checkpoint-every N`` atomically replaces the same path at absolute step
multiples of ``N`` and requires ``--checkpoint``.  Quasar writes a unique
same-directory temporary file, closes it collectively, and has rank zero
replace the committed path only after every rank reports success.  A failed
write leaves the previous committed checkpoint intact.

Restart may use a different rank count, GPU count, or tile shape:

.. code-block:: shell

   mpiexec -n 1 python -m quasar.pic.cli run examples/two_stream/input.yaml \
     --devices 0,1 --decomposition 1x2 \
     --restart /scratch/$USER/two-stream.h5 \
     --checkpoint /scratch/$USER/two-stream-next.h5

The deck's ``time.steps`` and optional end time are absolute termination
targets, not additional work after the checkpoint.  Restart preserves the
physical mesh, precision, geometry, units, boundaries, species, backgrounds,
numerical schemes, and timestep policy; changing any of them is rejected.  An
``auto`` timestep must remain ``auto``, and a fixed timestep must retain the same
value.  ``time.steps`` and the optional end time are not part of that timestep
identity, so they remain valid absolute termination-policy changes.  Only
placement, decomposition, diagnostics/checkpoint policy, and absolute
termination targets may otherwise change.  PIC restores the committed particle
and field initialization state rather than reseeding, so an explicit ``--seed``
cannot be combined with ``--restart``.

Use the Python API
------------------

The high-level Python entry points accept the same placement and I/O policy:

.. code-block:: python

   from quasar import distributed
   from quasar.mhd import run

   options = distributed.RunOptions(
       devices="auto",
       decomposition=(2, 2),
       transport="auto",
       diagnostics_layout="sharded",
       checkpoint="/scratch/user/orszag-tang.h5",
       checkpoint_every=50,
   )
   result = run("examples/orszag_tang/input.yaml", options=options)
   print(result.final_step, result.final_time_s, result.output_path)
   print(result.telemetry)

Omit ``options`` to retain the serial path.  Existing ``prepare_run``
signatures and return values remain serial and unchanged.  A distributed
``RunResult`` identifies its diagnostics and checkpoint artifacts and includes
wall/phase timing plus the requested and selected transport in ``telemetry``.

Failure and resource rules
--------------------------

Validation and runtime errors are converted to one collective decision before
the next MPI or HDF5 phase.  MHD recoverable step failures roll every tile back
and use one common retry decision.  A PIC failure after particle/field mutation
poisons the distributed session; only collective ``close()`` remains legal,
and recovery starts from the last committed checkpoint.

Quasar v1 requires the same positive GPU count on every rank and does not
support GPU oversubscription, dynamic load balancing, mesh regridding on
restart, process-failure recovery, coupled PIC-MHD evolution, or
topology-independent bitwise results.  The launcher remains responsible for
CPU and NUMA binding.  See ``examples/distributed/`` for multi-node scheduler
commands.

CTest GPU allocation
--------------------

Distributed tests declare CTest ``RESOURCE_GROUPS``. The distributed test
presets load ``tests/ctest_gpus.json`` and the common launcher maps each
allocation to ``HIP_VISIBLE_DEVICES``, preventing parallel tests from sharing a
GPU. The checked-in specification describes an eight-GPU ROCm node. On a node
with a different visible inventory, supply a site-local file explicitly:

.. code-block:: shell

   ctest --test-dir build/hip-gfx942-distributed-release \
     --resource-spec-file /path/to/ctest-gpus.json \
     --output-on-failure

Running CTest directly without a resource specification does not schedule GPU
resources; use a distributed preset or pass ``--resource-spec-file``.
