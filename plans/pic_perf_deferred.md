# Deferred PIC performance items (P2, P4)

Two performance optimizations from the deep-review fix plan are intentionally
**deferred**, not dropped. Both touch the same kernel-launch / async-ordering
path implicated in the field-BC heisenbug (see `field_bc_heisenbug.md`), so
landing them now would risk re-destabilizing a green suite for a pure
throughput gain with no correctness benefit.

## Delivered in Phase 4 (for context)
- Dead `diagnostics_hip.hip` (no-op, no callers) removed.
- Real particle compaction (`particle_compact_hip.hip`): atomic-scatter stream
  compaction of the alive particles to the front of the arrays, run on a
  64-step cadence in `EmPic2D3V::step()` only when an absorbing boundary exists.
- Device-side alive-count reduction (`launch_pic_alive_count`), used by both the
  C++ `alive_count` diagnostic and the Python CLI `_record_scalars` (replaces a
  full 7-array `to_host()` per logged step).
- `__restrict__` on read-only pointers in the push, deposit, and FDTD kernels.

## P2 — fused boundary kernel (deferred)
Replace the up-to-4 per-side particle-BC launches with a single kernel handling
all four sides + per-side kind in one pass. **Risk:** the particle-BC dispatch
is exactly the path that currently uses the heisenbug workaround (direct kernel
calls with `grid_` instead of `particle_bcs_[side]->apply`). Fusing it rewrites
that launch surface; do it *after* the heisenbug is root-caused so the two
changes don't confound each other.

## P4 — persistent hipStream_t (deferred)
Add `hipStream_t stream_` to `EmPic2D3V` (create in ctor, destroy in dtor),
thread it through every launch + async copy to replace null-stream
serialization. **Risk:** the heisenbug's leading hypothesis is a
kernarg-marshaling / async-ordering issue. A non-null persistent stream changes
ordering semantics relative to the synchronous `hipMemcpy` readbacks in
`to_host()` / diagnostics — i.e. it perturbs precisely the variable under
suspicion. Introduce it once the launch path is trusted.

## Re-enable order
1. Root-cause the field-BC heisenbug (`field_bc_heisenbug.md`).
2. Land P4 (persistent stream) with the BC path already trusted.
3. Land P2 (fused boundary kernel) on top.
