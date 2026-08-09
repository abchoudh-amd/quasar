if(NOT TARGET quasar_distributed)
  return()
endif()

set(_quasar_ctest_gpu_launcher
    "${PROJECT_SOURCE_DIR}/tests/run_with_ctest_gpus.sh")

foreach(_test_name IN ITEMS
    test_topology
    test_device_mapping
    test_collective_error
    test_worker_pool
    test_transport_policy
    test_mhd_halo
    test_checkpoint_metadata
    test_diagnostics_manifest)
  set(_target "quasar_distributed_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::distributed
  )
  if(_test_name STREQUAL "test_worker_pool")
    target_link_libraries(${_target} PRIVATE GTest::gtest)
    add_test(
      NAME "distributed_${_test_name}"
      COMMAND "${_quasar_ctest_gpu_launcher}" -- $<TARGET_FILE:${_target}>
    )
  else()
    target_link_libraries(${_target} PRIVATE GTest::gtest_main)
    add_test(NAME "distributed_${_test_name}" COMMAND ${_target})
  endif()
  set_tests_properties("distributed_${_test_name}" PROPERTIES
    LABELS "distributed;cpu"
    TIMEOUT 30
  )
  if(_test_name STREQUAL "test_worker_pool")
    set_tests_properties("distributed_${_test_name}" PROPERTIES
      LABELS "distributed;multigpu"
      RESOURCE_GROUPS "2,gpus:1"
      SKIP_RETURN_CODE 77
    )
  endif()
endforeach()

add_test(
  NAME distributed_test_ctest_gpu_launcher
  COMMAND bash
          "${CMAKE_CURRENT_SOURCE_DIR}/test_ctest_gpu_launcher.sh"
          "${_quasar_ctest_gpu_launcher}"
)
set_tests_properties(distributed_test_ctest_gpu_launcher PROPERTIES
  LABELS "distributed;cpu"
  TIMEOUT 30
)

add_executable(quasar_distributed_test_mpi_runtime test_mpi_runtime.cpp)
target_link_libraries(quasar_distributed_test_mpi_runtime
  PRIVATE
    quasar::distributed
)

add_test(
  NAME distributed_test_mpi_runtime
  COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_mpi_runtime>
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_mpi_runtime PROPERTIES
  LABELS "distributed;multirank"
  TIMEOUT 30
  SKIP_RETURN_CODE 77
)

add_executable(quasar_distributed_test_runtime_session
  test_runtime_session.cpp)
target_link_libraries(quasar_distributed_test_runtime_session
  PRIVATE
    quasar::distributed
    MPI::MPI_CXX
)

foreach(_runtime_session_mode IN ITEMS lifecycle abandoned)
  add_test(
    NAME "distributed_test_runtime_session_${_runtime_session_mode}"
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:quasar_distributed_test_runtime_session>
      ${_runtime_session_mode}
      ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    "distributed_test_runtime_session_${_runtime_session_mode}" PROPERTIES
    LABELS "distributed;multirank;session"
    TIMEOUT 30
    SKIP_RETURN_CODE 77
  )
endforeach()

add_test(
  NAME distributed_test_runtime_session_close_failure
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_runtime_session>
          owned-close-failure
)
set_tests_properties(distributed_test_runtime_session_close_failure PROPERTIES
  LABELS "distributed;multigpu;session"
  RESOURCE_GROUPS "gpus:1"
  TIMEOUT 30
  SKIP_RETURN_CODE 77
)

add_executable(quasar_distributed_test_transport_mpi test_transport_mpi.cpp)
target_link_libraries(quasar_distributed_test_transport_mpi
  PRIVATE
    quasar::distributed
)

add_test(
  NAME distributed_test_transport_mpi
  COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_transport_mpi>
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_transport_mpi PROPERTIES
  LABELS "distributed;multirank"
  TIMEOUT 30
  SKIP_RETURN_CODE 77
)

add_executable(quasar_distributed_test_transport_device_mpi
  test_transport_device_mpi.cpp)
target_link_libraries(quasar_distributed_test_transport_device_mpi
  PRIVATE
    quasar::distributed
    MPI::MPI_CXX
)

foreach(_transport_mode IN ITEMS staged direct)
  add_test(
    NAME "distributed_test_transport_device_${_transport_mode}_mpi"
    COMMAND
      "${_quasar_ctest_gpu_launcher}" --
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:quasar_distributed_test_transport_device_mpi>
      ${_transport_mode}
      ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(
    "distributed_test_transport_device_${_transport_mode}_mpi" PROPERTIES
    LABELS "distributed;multigpu;multirank;transport"
    RESOURCE_GROUPS "2,gpus:1"
    TIMEOUT 30
    SKIP_RETURN_CODE 77
  )
endforeach()

add_executable(quasar_distributed_test_distributed_io test_distributed_io.cpp)
target_link_libraries(quasar_distributed_test_distributed_io
  PRIVATE
    quasar::distributed
    HDF5::HDF5
    MPI::MPI_CXX
)

add_test(
  NAME distributed_test_distributed_io
  COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_distributed_io>
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_distributed_io PROPERTIES
  LABELS "distributed;multirank"
  TIMEOUT 60
  SKIP_RETURN_CODE 77
)

add_executable(quasar_distributed_test_mhd_runtime test_mhd_runtime.cpp)
target_link_libraries(quasar_distributed_test_mhd_runtime
  PRIVATE
    quasar::distributed
    MPI::MPI_CXX
)

add_executable(quasar_distributed_test_mhd_device_halo
  test_mhd_device_halo.cpp)
target_link_libraries(quasar_distributed_test_mhd_device_halo
  PRIVATE
    quasar::distributed
    GTest::gtest_main
)

add_test(
  NAME distributed_test_mhd_device_halo
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_device_halo>
)
set_tests_properties(distributed_test_mhd_device_halo PROPERTIES
  LABELS "distributed;multigpu;mhd;halo"
  RESOURCE_GROUPS "gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 30
)

add_test(
  NAME distributed_test_mhd_runtime_single_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_runtime> single
)
set_tests_properties(distributed_test_mhd_runtime_single_gpu PROPERTIES
  LABELS "distributed;multigpu;mhd"
  RESOURCE_GROUPS "gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_mhd_runtime_multi_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_runtime> multi
)
set_tests_properties(distributed_test_mhd_runtime_multi_gpu PROPERTIES
  LABELS "distributed;multigpu;mhd"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_mhd_runtime_four_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_runtime> quad
)
set_tests_properties(distributed_test_mhd_runtime_four_gpu PROPERTIES
  LABELS "distributed;multigpu;mhd"
  RESOURCE_GROUPS "4,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_mhd_runtime_multirank
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_mhd_runtime> multirank
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_mhd_runtime_multirank PROPERTIES
  LABELS "distributed;multigpu;multirank;mhd"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_executable(quasar_distributed_test_mhd_checkpoint
  test_mhd_checkpoint.cpp)
target_link_libraries(quasar_distributed_test_mhd_checkpoint
  PRIVATE
    quasar::distributed
    HDF5::HDF5
    MPI::MPI_CXX
)

add_test(
  NAME distributed_test_mhd_checkpoint_single_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint> single
)
set_tests_properties(distributed_test_mhd_checkpoint_single_gpu PROPERTIES
  LABELS "distributed;multigpu;mhd;checkpoint;restart"
  RESOURCE_GROUPS "gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_mhd_checkpoint_repartition
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint> repartition
)
set_tests_properties(distributed_test_mhd_checkpoint_repartition PROPERTIES
  LABELS "distributed;multigpu;mhd;checkpoint;restart"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_mhd_checkpoint_multirank
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint>
    multirank
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_mhd_checkpoint_multirank PROPERTIES
  LABELS "distributed;multigpu;multirank;mhd;checkpoint;restart"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

set(_mhd_cross_process_checkpoint
    "${CMAKE_CURRENT_BINARY_DIR}/mhd-cross-process-restart.h5")
add_test(
  NAME distributed_test_mhd_checkpoint_cross_process_write
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint>
          write-cross "${_mhd_cross_process_checkpoint}"
)
set_tests_properties(distributed_test_mhd_checkpoint_cross_process_write PROPERTIES
  LABELS "distributed;multigpu;mhd;checkpoint;restart"
  RESOURCE_GROUPS "gpus:1"
  FIXTURES_SETUP mhd_checkpoint_cross_process
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)
add_test(
  NAME distributed_test_mhd_checkpoint_cross_process_read
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint>
    read-cross "${_mhd_cross_process_checkpoint}"
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_mhd_checkpoint_cross_process_read PROPERTIES
  LABELS "distributed;multigpu;multirank;mhd;checkpoint;restart"
  RESOURCE_GROUPS "2,gpus:1"
  FIXTURES_REQUIRED mhd_checkpoint_cross_process
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

set(_mhd_cross_placement_checkpoint
    "${CMAKE_CURRENT_BINARY_DIR}/mhd-cross-placement-restart.h5")
add_test(
  NAME distributed_test_mhd_checkpoint_two_rank_four_gpu_write
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint>
    write-cross-four "${_mhd_cross_placement_checkpoint}"
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(
  distributed_test_mhd_checkpoint_two_rank_four_gpu_write PROPERTIES
  LABELS "distributed;multigpu;multirank;mhd;checkpoint;restart"
  RESOURCE_GROUPS "4,gpus:1"
  FIXTURES_SETUP mhd_checkpoint_cross_placement
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)
add_test(
  NAME distributed_test_mhd_checkpoint_one_rank_four_gpu_read
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_mhd_checkpoint>
          read-cross-four "${_mhd_cross_placement_checkpoint}"
)
set_tests_properties(
  distributed_test_mhd_checkpoint_one_rank_four_gpu_read PROPERTIES
  LABELS "distributed;multigpu;mhd;checkpoint;restart"
  RESOURCE_GROUPS "4,gpus:1"
  FIXTURES_REQUIRED mhd_checkpoint_cross_placement
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_executable(quasar_distributed_test_pic_runtime test_pic_runtime.cpp)
target_link_libraries(quasar_distributed_test_pic_runtime
  PRIVATE
    quasar::distributed
    MPI::MPI_CXX
)

add_test(
  NAME distributed_test_pic_runtime_single_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_pic_runtime> single
)
set_tests_properties(distributed_test_pic_runtime_single_gpu PROPERTIES
  LABELS "distributed;multigpu;pic"
  RESOURCE_GROUPS "gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_pic_runtime_multi_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_pic_runtime> multi
)
set_tests_properties(distributed_test_pic_runtime_multi_gpu PROPERTIES
  LABELS "distributed;multigpu;pic"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_pic_runtime_four_gpu_diagonal
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_pic_runtime> quad
)
set_tests_properties(distributed_test_pic_runtime_four_gpu_diagonal PROPERTIES
  LABELS "distributed;multigpu;pic"
  RESOURCE_GROUPS "4,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_pic_runtime_multirank
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_pic_runtime> multirank
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_pic_runtime_multirank PROPERTIES
  LABELS "distributed;multirank;multigpu;pic"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_test(
  NAME distributed_test_pic_runtime_local_shards_multirank
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_pic_runtime> local-shards
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(
  distributed_test_pic_runtime_local_shards_multirank PROPERTIES
  LABELS "distributed;multirank;multigpu;pic;diagnostics"
  RESOURCE_GROUPS "4,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 60
)

add_executable(quasar_distributed_test_pic_checkpoint
  test_pic_checkpoint.cpp)
target_link_libraries(quasar_distributed_test_pic_checkpoint
  PRIVATE
    quasar::distributed
    HDF5::HDF5
    MPI::MPI_CXX
)

add_test(
  NAME distributed_test_pic_checkpoint_single_gpu
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_pic_checkpoint> single
)
set_tests_properties(distributed_test_pic_checkpoint_single_gpu PROPERTIES
  LABELS "distributed;multigpu;pic;checkpoint;restart"
  RESOURCE_GROUPS "gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_pic_checkpoint_repartition
  COMMAND "${_quasar_ctest_gpu_launcher}" --
          $<TARGET_FILE:quasar_distributed_test_pic_checkpoint> repartition
)
set_tests_properties(distributed_test_pic_checkpoint_repartition PROPERTIES
  LABELS "distributed;multigpu;pic;checkpoint;restart"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_pic_checkpoint_multirank
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_pic_checkpoint>
    multirank
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(distributed_test_pic_checkpoint_multirank PROPERTIES
  LABELS "distributed;multigpu;multirank;pic;checkpoint;restart"
  RESOURCE_GROUPS "2,gpus:1"
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

set(_pic_cross_rank_checkpoint
  "${CMAKE_CURRENT_BINARY_DIR}/quasar-pic-cross-rank.h5")
add_test(
  NAME distributed_test_pic_checkpoint_two_rank_two_gpu_write
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:quasar_distributed_test_pic_checkpoint>
    cross-write
    "${_pic_cross_rank_checkpoint}"
    ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(
  distributed_test_pic_checkpoint_two_rank_two_gpu_write PROPERTIES
  LABELS "distributed;multigpu;multirank;pic;checkpoint;restart"
  RESOURCE_GROUPS "4,gpus:1"
  FIXTURES_SETUP pic_checkpoint_cross_rank
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)

add_test(
  NAME distributed_test_pic_checkpoint_one_rank_four_gpu_read
  COMMAND
    "${_quasar_ctest_gpu_launcher}" --
    $<TARGET_FILE:quasar_distributed_test_pic_checkpoint>
    cross-read
    "${_pic_cross_rank_checkpoint}"
)
set_tests_properties(
  distributed_test_pic_checkpoint_one_rank_four_gpu_read PROPERTIES
  LABELS "distributed;multigpu;pic;checkpoint;restart"
  RESOURCE_GROUPS "4,gpus:1"
  FIXTURES_REQUIRED pic_checkpoint_cross_rank
  SKIP_RETURN_CODE 77
  TIMEOUT 120
)
