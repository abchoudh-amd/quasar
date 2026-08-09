import unittest

import numpy as np

import quasar._distributed as native_distributed
import quasar.distributed as distributed
from quasar import mhd
from quasar import pic


def _ctest_exit_code(result):
    """Map an otherwise-successful resource skip to CTest's skip code."""

    if not result.wasSuccessful():
        return 1
    return 77 if result.skipped else 0


def _mhd_config(nx, ny, *, background=False):
    config = mhd.MhdConfig()
    config.grid = mhd.Grid2D(nx, ny, 1.0, 1.0, nghost=2)
    config.reconstruction = "muscl_minmod"
    boundary = mhd.MhdBoundarySpec()
    boundary.set_fluid_all("outflow")
    boundary.set_field_all("outflow")
    config.boundary = boundary
    if background:
        specification = config.background
        specification.enabled = True
        specification.profile = "uniform"
        specification.bx0 = 0.0
        specification.by0 = 0.0
        specification.bz0 = 0.0
        config.background = specification
    return config


def _mhd_state(nx, ny):
    return {
        "global_nx": nx,
        "global_ny": ny,
        "rho": np.ones((ny, nx), dtype=np.float64),
        "mx": np.zeros((ny, nx), dtype=np.float64),
        "my": np.zeros((ny, nx), dtype=np.float64),
        "mz": np.zeros((ny, nx), dtype=np.float64),
        "energy": np.full((ny, nx), 3.0, dtype=np.float64),
        "bx_face": np.full((ny, nx + 1), 0.5, dtype=np.float64),
        "by_face": np.full((ny + 1, nx), -0.25, dtype=np.float64),
        "bz_cell": np.full((ny, nx), 0.125, dtype=np.float64),
    }


def _mhd_background(nx, ny):
    return {
        "global_nx": nx,
        "global_ny": ny,
        "b0x_face": np.full((ny, nx + 1), 0.25, dtype=np.float64),
        "b0y_face": np.full((ny + 1, nx), -0.125, dtype=np.float64),
        "b0z_cell": np.full((ny, nx), 0.0625, dtype=np.float64),
    }


def _pic_config(nx, ny):
    config = pic.EmPicConfig()
    config.grid = pic.Grid2D(nx, ny, 1.0, 1.0, nghost=1)
    config.fdtd_order = 2
    config.shape = "cic"
    return config


def _pic_fields(nx, ny):
    return {
        "global_nx": nx,
        "global_ny": ny,
        "ex": np.zeros((ny, nx + 1), dtype=np.float64),
        "ey": np.zeros((ny + 1, nx), dtype=np.float64),
        "ez": np.zeros((ny, nx), dtype=np.float64),
        "bx": np.zeros((ny + 1, nx), dtype=np.float64),
        "by": np.zeros((ny, nx + 1), dtype=np.float64),
        "bz": np.zeros((ny + 1, nx + 1), dtype=np.float64),
    }


class NativeAvailabilityTests(unittest.TestCase):

    def test_foundation_reports_complete_python_orchestrated_runners(self):
        self.assertIs(native_distributed.foundation_available(), True)
        self.assertIs(distributed.foundation_available(), True)
        self.assertIs(native_distributed.pic_runtime_available(), True)
        self.assertIs(native_distributed.mhd_runtime_available(), True)
        self.assertIs(native_distributed.is_available(), True)
        self.assertIs(distributed.is_available(), True)
        self.assertIsNone(native_distributed.unavailable_reason())
        self.assertIsNone(distributed.unavailable_reason())
        self.assertIs(distributed.RuntimeSession,
                      native_distributed.RuntimeSession)
        # Physics orchestration is deliberately implemented in staged Python
        # modules over the collective-safe RuntimeSession surface.
        self.assertFalse(hasattr(native_distributed, "run_pic"))
        self.assertFalse(hasattr(native_distributed, "run_mhd"))


class CtestExitCodeTests(unittest.TestCase):

    def test_success_and_resource_skip_have_distinct_process_codes(self):
        success = unittest.TestResult()
        self.assertEqual(_ctest_exit_code(success), 0)

        skipped = unittest.TestResult()
        skipped.addSkip(self, "GPU resource unavailable")
        self.assertEqual(_ctest_exit_code(skipped), 77)


@unittest.skipUnless(
    native_distributed.visible_device_count() > 0,
    "no HIP device is visible",
)
class NativeRuntimeSessionTests(unittest.TestCase):

    def assert_transport_resolution(self, transport, requested):
        self.assertEqual(transport["requested"], requested)
        if requested == "auto":
            direct_available = (
                transport["direct_query_recognized"]
                and transport["direct_startup_probe"])
            expected = "direct" if direct_available else "staged"
        else:
            expected = requested
        self.assertEqual(transport["interprocess"], expected)

    def test_collective_session_maps_endpoints_and_selects_topology(self):
        with native_distributed.RuntimeSession() as session:
            self.assertGreaterEqual(session.rank, 0)
            self.assertGreaterEqual(session.size, 1)
            self.assertGreaterEqual(session.node_rank, 0)
            self.assertGreaterEqual(session.node_size, 1)
            with self.assertRaisesRegex(RuntimeError, "one live RuntimeSession"):
                native_distributed.RuntimeSession()

            session.barrier()
            session.collective_agree(
                "shared-policy", "native-agreement-success")
            if session.size > 1:
                with self.assertRaisesRegex(
                        RuntimeError, "rank policies differ") as disagreement:
                    session.collective_agree(
                        f"rank-{session.rank}",
                        "native-agreement-mismatch",
                        "rank policies differ")
                self.assertIn(
                    "native-agreement-mismatch", str(disagreement.exception))

            malformed_selection = (
                [False] if session.rank == 0 else "auto")
            with self.assertRaisesRegex(RuntimeError, "python-device-input"):
                session.configure_devices(malformed_selection)

            valid_devices = [
                {
                    "ordinal": 0,
                    "uuid": f"test-rank-{session.rank}-gpu-0",
                    "pci_bus_id": "",
                },
            ]
            invalid_devices = (
                [{"ordinal": "invalid", "uuid": "bad"}]
                if session.rank == 0 else valid_devices)
            with self.assertRaisesRegex(RuntimeError, "python-endpoint-input"):
                session.configure_owned_devices(invalid_devices)

            endpoints = session.configure_owned_devices(valid_devices)
            self.assertEqual(len(endpoints), session.size)
            self.assertEqual(
                [endpoint["index"] for endpoint in endpoints],
                list(range(session.size)))

            global_nx = max(16, 8 * session.size)
            with self.assertRaises(TypeError):
                session.select_topology(global_nx, 16, "auto")
            if session.size > 1:
                conflicting_shape = (
                    (session.size, 1) if session.rank == 0
                    else (1, session.size))
                with self.assertRaisesRegex(
                        RuntimeError, "different topology arguments"):
                    session.select_topology(
                        global_nx, 16, conflicting_shape, 2)

            topology = session.select_topology(global_nx, 16, "auto", 2)
            self.assertEqual(
                topology["decomposition"][0]
                * topology["decomposition"][1],
                session.size)
            self.assertEqual(len(topology["tiles"]), session.size)
            self.assertEqual(session.telemetry["barriers"], 1)
            self.assertEqual(session.telemetry["endpoint_configurations"], 1)
            self.assertEqual(session.telemetry["topology_selections"], 1)

            # The endpoint identity is deliberately rank-unique and the ordinal
            # follows the node-local rank, so the two-rank CTest uses disjoint
            # GPUs without relying on mpi4py for rank/device coordination.
            devices = [{
                "ordinal": session.node_rank,
                "uuid": f"native-mhd-rank-{session.rank}",
                "pci_bus_id": "",
            }]
            try:
                session.configure_owned_devices(devices)
                nx = max(8, 8 * session.size)
                ny = 8
                session.select_topology(nx, ny, (session.size, 1), 2)
            except RuntimeError as error:
                self.skipTest(f"distributed MHD device setup unavailable: {error}")

            # A restart failure after candidate construction must close and
            # unpublish that candidate so a fresh seed can start immediately.
            missing_checkpoint = "/dev/null/quasar-missing-checkpoint.h5"
            with self.assertRaises(RuntimeError):
                session.restart_mhd(
                    _mhd_config(nx, ny), missing_checkpoint, "normalized",
                    None, "auto")

            state = _mhd_state(nx, ny)
            malformed_state = dict(state)
            if session.rank == 0:
                malformed_state.pop("rho")
            with self.assertRaisesRegex(
                    RuntimeError, "python-mhd-input") as state_error:
                session.start_mhd(
                    _mhd_config(nx, ny), malformed_state)
            self.assertIn("on rank 0", str(state_error.exception))

            # This seed is fully parseable and internally valid on every rank,
            # so the native runtime and its worker pool are constructed before
            # rank zero's topology mismatch is rejected collectively.  The
            # unpublished candidate must clean up without consuming the
            # process-wide session slot or topology.
            mismatched_state = state
            if session.rank == 0:
                mismatched_state = _mhd_state(nx + 1, ny)
            with self.assertRaisesRegex(
                    RuntimeError, "mhd-seed-validate") as seed_error:
                session.start_mhd(_mhd_config(nx, ny), mismatched_state)
            self.assertIn("on rank 0", str(seed_error.exception))

            # Both a parse failure and an unpublished native-runtime failure
            # leave the same collective session immediately reusable.
            try:
                session.start_mhd(_mhd_config(nx, ny), state)
            except RuntimeError as error:
                self.skipTest(f"distributed MHD runtime unavailable: {error}")
            with self.assertRaisesRegex(
                    RuntimeError, "python-endpoint-runtime"):
                session.configure_owned_devices(devices)
            with self.assertRaisesRegex(
                    RuntimeError, "python-topology-runtime"):
                session.select_topology(nx, ny, (session.size, 1), 2)
            # Both reconfiguration requests are rejected before mutation; the
            # active runtime and its mapping/topology remain usable.
            mhd_cfl = session.mhd_cfl_limit()
            self.assertGreater(mhd_cfl, 0.0)
            invalid_mhd_dt = (
                "not-a-timestep" if session.rank == 0 else 0.1 * mhd_cfl)
            with self.assertRaisesRegex(
                    RuntimeError, "python-mhd-step-input"):
                session.mhd_step(invalid_mhd_dt)
            if session.size > 1:
                mismatched_mhd_dt = (
                    0.05 * mhd_cfl if session.rank == 0
                    else 0.1 * mhd_cfl)
                with self.assertRaisesRegex(
                        RuntimeError, "python-mhd-step-agreement"):
                    session.mhd_step(mismatched_mhd_dt)
            session.mhd_step(0.1 * mhd_cfl)
            mhd_telemetry = session.telemetry["mhd"]
            self.assert_transport_resolution(
                mhd_telemetry["transport"], "auto")
            session.close_mhd()

            # A cleanup exception must not discard the unpublished candidate.
            # The test hook simulates that one-shot close failure: another
            # start is rejected until explicit collective close releases the
            # retained candidate, after which the session is reusable.
            cleanup_hook = getattr(
                session,
                "_inject_candidate_cleanup_failure_for_testing", None)
            if cleanup_hook is not None:
                cleanup_hook(True)
                with self.assertRaisesRegex(
                        RuntimeError, "mhd-seed-validate"):
                    session.start_mhd(
                        _mhd_config(nx, ny), mismatched_state)
                with self.assertRaisesRegex(
                        RuntimeError, "python-mhd-start"):
                    session.start_mhd(_mhd_config(nx, ny), state)
                session.close_mhd()
                session.start_mhd(_mhd_config(nx, ny), state)
                self.assertGreater(session.mhd_cfl_limit(), 0.0)
                session.close_mhd()

            fields = _pic_fields(nx, ny)
            with self.assertRaises(RuntimeError):
                session.restart_pic(
                    _pic_config(nx, ny), missing_checkpoint, "normalized",
                    [], "auto")
            nonfinite_fields = {
                name: (np.array(values, copy=True)
                       if isinstance(values, np.ndarray) else values)
                for name, values in fields.items()
            }
            if session.rank == 0:
                nonfinite_fields["ex"][0, 0] = np.nan
            with self.assertRaisesRegex(
                    RuntimeError, "pic-seed-validate") as pic_seed_error:
                session.start_pic(
                    _pic_config(nx, ny), nonfinite_fields, None, [],
                    transport="auto")
            self.assertIn("on rank 0", str(pic_seed_error.exception))

            session.start_pic(
                _pic_config(nx, ny), fields, None, [],
                transport="auto")
            pic_cfl = session.pic_cfl_limit()
            self.assertGreater(pic_cfl, 0.0)
            invalid_pic_dt = (
                "not-a-timestep" if session.rank == 0 else 0.1 * pic_cfl)
            with self.assertRaisesRegex(
                    RuntimeError, "python-pic-step-input"):
                session.pic_step(invalid_pic_dt)
            if session.size > 1:
                mismatched_pic_dt = (
                    0.05 * pic_cfl if session.rank == 0
                    else 0.1 * pic_cfl)
                with self.assertRaisesRegex(
                        RuntimeError, "python-pic-step-agreement"):
                    session.pic_step(mismatched_pic_dt)
            session.pic_step(0.1 * pic_cfl)
            pic_telemetry = session.telemetry["pic"]
            self.assert_transport_resolution(
                pic_telemetry["transport"], "auto")
            self.assertEqual(
                pic_telemetry["transport_bytes"],
                pic_telemetry["transport"]["bytes"])
            self.assertEqual(
                pic_telemetry["checkpoint_local_lattice_writes"], 0)
            self.assertEqual(
                pic_telemetry["checkpoint_local_lattice_reads"], 0)
            self.assertEqual(
                pic_telemetry[
                    "checkpoint_global_lattice_materializations"], 0)
            session.close_pic()

            if cleanup_hook is not None:
                cleanup_hook(True)
                with self.assertRaisesRegex(
                        RuntimeError, "pic-seed-validate"):
                    session.start_pic(
                        _pic_config(nx, ny), nonfinite_fields, None, [],
                        transport="auto")
                with self.assertRaisesRegex(
                        RuntimeError, "python-pic-start"):
                    session.start_pic(
                        _pic_config(nx, ny), fields, None, [],
                        transport="auto")
                session.close_pic()
                session.start_pic(
                    _pic_config(nx, ny), fields, None, [],
                    transport="auto")
                self.assertGreater(session.pic_cfl_limit(), 0.0)
                session.close_pic()

            background = _mhd_background(nx, ny)
            malformed_background = dict(background)
            if session.rank == 0:
                malformed_background.pop("b0z_cell")
            with self.assertRaisesRegex(
                    RuntimeError, "python-mhd-input") as background_error:
                session.start_mhd(
                    _mhd_config(nx, ny, background=True), state,
                    malformed_background)
            self.assertIn("on rank 0", str(background_error.exception))

            session.start_mhd(
                _mhd_config(nx, ny, background=True), state, background)
            self.assertGreater(session.mhd_cfl_limit(), 0.0)
            session.close_mhd()

        self.assertIs(session.closed, True)
        session.close()


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    raise SystemExit(_ctest_exit_code(program.result))
