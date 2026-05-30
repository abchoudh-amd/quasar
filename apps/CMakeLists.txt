# No C++ apps are built today. Both vertical slices are driven from Python:
#   - magnetostatics: `python -m quasar.coil.cli run <deck.yaml>`
#   - PIC:            `python -m quasar.pic.cli  run <deck.yaml>`
# The thin C++ entry points (apps/*/main.cpp) were placeholder stubs that only
# parsed nothing and printed a banner; they were removed in favor of the Python
# drivers. Add a real deck-driver target here if a standalone C++ binary is ever
# needed.
