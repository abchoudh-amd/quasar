Adding a PIC particle pusher
============================

Particle pushers implement ``quasar::numerics::IParticlePusher``. A pusher is
responsible for gathering fields at particle positions, updating velocity, and
advancing position. Existing code provides ``BorisPusher<ShapeOrder>`` as the
default non-relativistic pusher.

To add a pusher, declare a concrete class in
``include/quasar/numerics/particle_pusher.hpp``, implement its launch wrapper
in ``src/physics/pic`` or ``src/backend/hip/pic``, and register it under a
deck-facing name with ``QUASAR_REGISTER_PUSHER("<name>", <Class>)`` (mirroring
the field-evaluator, boundary, and current-filter guides). The PIC solver then
builds it through the registry by that name, so no builder edit is needed.
