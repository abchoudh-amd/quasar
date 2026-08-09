# Distributed PIC and MHD launcher examples

These scripts launch one shared Quasar simulation across two nodes, one MPI
rank per node, and two GPUs per rank. The directory includes small PIC and MHD
decks so the launchers are self-contained and follow the same deck + README +
integration-test convention as the other examples. Run them from the repository
root after building the distributed Python extension.

Set `QUASAR_BUILD_DIR` to the configured build directory. The scheduler must
make two GPUs visible to each task; Quasar's `--devices auto` validates physical
device ownership and constructs four rank-ordered GPU endpoints. The example
`2x2` decomposition therefore covers one global mesh with four tiles.

```bash
export QUASAR_BUILD_DIR="$PWD/build/hip-gfx950-distributed-release"
sbatch examples/distributed/run_pic_multinode.sbatch
sbatch examples/distributed/run_mhd_multinode.sbatch
```

Override `QUASAR_PIC_DECK`, `QUASAR_MHD_DECK`, or `QUASAR_CHECKPOINT_DIR` in
the submission environment to select another deck or scratch location. The
physics YAML remains independent of rank and GPU placement.

The shipped decks run only eight steps and are intended as launcher smoke
cases. Use a production deck and a scheduler-appropriate checkpoint cadence for
science runs.

The site-specific partition, account, and wall-time directives are deliberately
omitted. Add the directives required by your cluster before submission.
