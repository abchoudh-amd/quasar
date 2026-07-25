# QuasarLaunchParams.cmake
#
# Helper for setting per-gfx-target Biot-Savart kernel launch parameters
# (TILE_SEGMENTS and BLOCK_SIZE) as compile definitions on a HIP target.
#
# The values are picked from CMAKE_HIP_ARCHITECTURES so that both the host
# launcher (kernel<<<grid, block, ...>>>) and the device kernel itself
# (__shared__ Real s_ax[kTileSegments]) see the same compile-time constants.
# Multi-target HIP builds use the first-listed architecture, since the same
# kernel object code is launched on whichever device is visible at runtime.
#
# Tuning data: the values are baseline starting points sized to fit comfortably
# in LDS for each architecture family. Refinement against the actual
# micro-benchmark sweep lives in benchmarks/micro/biot_savart_bench.cpp.

include_guard(GLOBAL)

function(quasar_set_launch_params target)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(_primary_arch "gfx942")
  else()
    list(GET CMAKE_HIP_ARCHITECTURES 0 _primary_arch)
  endif()

  # Strip optional feature flags such as gfx90a:sramecc+:xnack-.
  string(REGEX REPLACE "([^:]+):.*" "\\1" _arch "${_primary_arch}")

  set(_tile 64)
  set(_block 128)
  set(_family "unknown")

  if(_arch STREQUAL "gfx942")
    # MI300 family (CDNA3): 64KB LDS, wavefront 64, large CU count.
    set(_tile 128)
    set(_block 256)
    set(_family "gfx94x (CDNA3)")
  elseif(_arch STREQUAL "gfx950")
    # MI350 family (CDNA4): retain the same conservative launch baseline.
    set(_tile 128)
    set(_block 256)
    set(_family "gfx95x (CDNA4)")
  elseif(_arch STREQUAL "gfx90a" OR _arch STREQUAL "gfx908")
    # MI200 family: 64KB LDS, wavefront 64.
    set(_tile 128)
    set(_block 256)
    set(_family "gfx9x (CDNA1/CDNA2)")
  elseif(_arch MATCHES "^gfx11[0-9][0-9]$")
    # RDNA3 consumer: wavefront 32, smaller blocks pack more wavefronts/CU.
    set(_tile 64)
    set(_block 128)
    set(_family "gfx11xx (RDNA3)")
  elseif(_arch MATCHES "^gfx12[0-9][0-9]$")
    # RDNA4: starting point inherits RDNA3.
    set(_tile 64)
    set(_block 128)
    set(_family "gfx12xx (RDNA4)")
  else()
    message(STATUS
      "quasar_set_launch_params: unrecognized gfx target '${_arch}'; "
      "using conservative defaults TILE=${_tile} BLOCK=${_block}.")
  endif()

  message(STATUS
    "quasar launch params for ${target} (arch ${_arch}, ${_family}): "
    "TILE_SEGMENTS=${_tile} BLOCK_SIZE=${_block}")

  target_compile_definitions(${target} PRIVATE
    QUASAR_BS_TILE_SEGMENTS=${_tile}
    QUASAR_BS_BLOCK_SIZE=${_block}
  )
endfunction()
