import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from quasar import _core
from quasar.pic._units import Units
from quasar.pic.cli import (
    _apply_external_field,
    _build_parser,
    _flatten_for_npz,
    _macro_weight,
    _make_solver,
    _seed_fields,
    _seed_species,
    _snapshot,
    _run_loop,
    prepare_run,
)
from quasar.pic.io import (
    BoundaryConfig,
    Diagnostics,
    Domain,
    ExternalField,
    Fields,
    FieldsInitial,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
    VelocityPerturbation,
)


class _RecordingSolver:
    """Captures seed_field(component, values) calls without a GPU."""

    def __init__(self):
        self.calls = []

    def seed_field(self, component, values):
        self.calls.append((component, np.asarray(values)))


def _fields_deck(initial: FieldsInitial) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                         initial=SpeciesInitial())],
        fields=Fields(initial=initial),
        units="normalized",
    )


class SeedFieldsTests(unittest.TestCase):

    def test_cylindrical_bessel_seed_initializes_matching_bphi_half_step(self):
        for order in (2, 4):
            with self.subTest(order=order):
                deck = _fields_deck(FieldsInitial(
                    type="seed_perturbation", component="Ey",
                    amplitude=1.25, mode=(1, 0)))
                deck.geometry = "cylindrical"
                deck.domain = Domain(
                    nx=8, ny=5, lx_m=2.0, ly_m=1.0, origin_x_m=0.0)
                deck.numerics = Numerics(fdtd_order=order, shape="cic")
                deck.boundary = BoundaryConfig(
                    particle=("axis", "specular", "specular", "specular"),
                    field=("axis", "pec", "pec", "pec"))
                deck.validate()

                dt = 0.0125
                solver = _RecordingSolver()
                _seed_fields(solver, deck, dt=dt, units=Units(deck))
                seeded = dict(solver.calls)
                self.assertEqual(seeded.keys(), {"ey", "bz"})

                nx, ny = deck.domain.nx, deck.domain.ny
                g = _core.pic.required_nghost(order)
                pitch, height = nx + 2 * g, ny + 2 * g
                ey = seeded["ey"].reshape(height, pitch)[
                    g:g + ny + 1, g:g + nx]
                bphi_minus = seeded["bz"].reshape(height, pitch)[
                    g:g + ny + 1, g:g + nx + 1]
                np.testing.assert_allclose(
                    ey, np.broadcast_to(ey[0], ey.shape), rtol=0.0, atol=0.0)
                np.testing.assert_allclose(
                    bphi_minus, np.broadcast_to(bphi_minus[0], bphi_minus.shape),
                    rtol=0.0, atol=0.0)

                row = ey[0]

                def sample(cell):
                    sign = 1.0
                    while cell < 0 or cell >= nx:
                        if cell < 0:
                            cell = -cell - 1
                        else:
                            cell = 2 * nx - cell - 1
                            sign = -sign
                    return sign * row[cell]

                derivative = np.empty(nx + 1)
                for face in range(nx + 1):
                    if order == 4:
                        derivative[face] = (
                            (9.0 / 8.0) * (sample(face) - sample(face - 1))
                            - (1.0 / 24.0)
                            * (sample(face + 1) - sample(face - 2))) / 0.25
                    else:
                        derivative[face] = (
                            sample(face) - sample(face - 1)) / 0.25

                # One Faraday update takes Bphi^{-1/2} to Bphi^{+1/2}; a
                # standing mode at its E maximum has exact half-step antisymmetry.
                bphi_plus = bphi_minus + dt * derivative[np.newaxis, :]
                np.testing.assert_allclose(
                    bphi_plus, -bphi_minus, rtol=2.0e-14, atol=2.0e-15)
                np.testing.assert_allclose(bphi_minus[:, 0], 0.0, atol=0.0)
                self.assertGreater(float(np.max(np.abs(bphi_minus))), 0.0)

    def test_tm_cavity_seeds_discrete_eigenvector_at_leapfrog_times(self):
        amp = 2.5
        deck = _fields_deck(FieldsInitial(
            type="seed_tm_cavity", component="Ez", amplitude=amp,
            mode=(2, 3)))
        deck.domain = Domain(nx=8, ny=6, lx_m=2.0, ly_m=1.5)
        deck.numerics = Numerics(fdtd_order=4, shape="cic")
        deck.boundary = BoundaryConfig(
            particle=("specular",) * 4, field=("pec",) * 4)
        dt = 0.025
        deck.validate()

        solver = _RecordingSolver()
        _seed_fields(solver, deck, dt=dt, units=Units(deck))
        seeded = {component: values for component, values in solver.calls}
        self.assertEqual(seeded.keys(), {"ez", "bx", "by"})

        nx, ny = deck.domain.nx, deck.domain.ny
        g = _core.pic.required_nghost(deck.numerics.fdtd_order)
        pitch, height = nx + 2 * g, ny + 2 * g
        padded = {
            component: values.reshape(height, pitch)
            for component, values in seeded.items()
        }
        ez = padded["ez"][g:g + ny, g:g + nx]
        bx = padded["bx"][g:g + ny + 1, g:g + nx]
        by = padded["by"][g:g + ny, g:g + nx + 1]

        mx, my = deck.fields.initial.mode
        sx = np.sin(np.pi * mx * (np.arange(nx) + 0.5) / nx)
        sy = np.sin(np.pi * my * (np.arange(ny) + 0.5) / ny)
        cx = np.cos(np.pi * mx * np.arange(nx + 1) / nx)
        cy = np.cos(np.pi * my * np.arange(ny + 1) / ny)
        dx, dy = deck.domain.lx_m / nx, deck.domain.ly_m / ny

        def symbol(mode, count, spacing):
            theta = np.pi * mode / (2.0 * count)
            return ((9.0 / 4.0) * np.sin(theta)
                    - (1.0 / 12.0) * np.sin(3.0 * theta)) / spacing

        kx = symbol(mx, nx, dx)
        ky = symbol(my, ny, dy)
        kappa = np.hypot(kx, ky)
        omega_dt = 2.0 * np.arcsin(0.5 * dt * kappa)
        half_time = np.sin(0.5 * omega_dt)

        np.testing.assert_allclose(
            ez, amp * np.multiply.outer(sy, sx), rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(
            bx, amp * (ky / kappa) * half_time * np.multiply.outer(cy, sx),
            rtol=2.0e-15, atol=2.0e-15)
        np.testing.assert_allclose(
            by, -amp * (kx / kappa) * half_time * np.multiply.outer(sy, cx),
            rtol=2.0e-15, atol=2.0e-15)

        # No padded/ghost slot is accidentally treated as a physical field DOF.
        self.assertTrue(np.all(padded["ez"][:g] == 0.0))
        self.assertTrue(np.all(padded["ez"][:, :g] == 0.0))
        self.assertTrue(np.all(padded["bx"][g + ny + 1:] == 0.0))
        self.assertTrue(np.all(padded["by"][:, g + nx + 1:] == 0.0))

    def test_tm_cavity_order2_uses_second_order_dispersion(self):
        deck = _fields_deck(FieldsInitial(
            type="seed_tm_cavity", component="Ez", amplitude=1.0,
            mode=(1, 1)))
        deck.boundary = BoundaryConfig(
            particle=("specular",) * 4, field=("pec",) * 4)
        dt = 0.04
        solver = _RecordingSolver()
        _seed_fields(solver, deck, dt=dt, units=Units(deck))
        seeded = dict(solver.calls)

        nx = ny = 8
        g = 1
        bx = seeded["bx"].reshape(ny + 2 * g, nx + 2 * g)[
            g:g + ny + 1, g:g + nx]
        theta = np.pi / (2.0 * nx)
        modified_k = 2.0 * np.sin(theta) / (1.0 / nx)
        omega_dt = 2.0 * np.arcsin(
            0.5 * dt * np.hypot(modified_k, modified_k))
        sx = np.sin(np.pi * (np.arange(nx) + 0.5) / nx)
        cy = np.cos(np.pi * np.arange(ny + 1) / ny)
        expected_bx = (np.sin(0.5 * omega_dt) / np.sqrt(2.0)
                       * np.multiply.outer(cy, sx))
        np.testing.assert_allclose(
            bx, expected_bx, rtol=2.0e-15, atol=2.0e-15)

    def test_seed_em_wave_rejects_nonzero_my(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Ez",
                                          mode=(1, 1)))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_seed_em_wave_rejects_unsupported_component(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Bz",
                                          mode=(1, 0)))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_unsupported_initial_type_raises(self):
        deck = _fields_deck(FieldsInitial(type="seed_blastwave", component="Ex"))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_seed_em_wave_ez_seeds_ez_and_by(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Ez",
                                          mode=(1, 0)))
        solver = _RecordingSolver()
        _seed_fields(solver, deck)
        seeded = {c for c, _ in solver.calls}
        self.assertEqual(seeded, {"ez", "by"})

    def test_si_em_wave_phase_matches_equivalent_normalized_deck(self):
        initial_si = FieldsInitial(
            type="seed_em_wave", component="Ez", amplitude=2.5,
            mode=(2, 0))
        si_deck = _fields_deck(initial_si)
        si_deck.units = "SI"
        si_deck.numerics = Numerics(fdtd_order=4, shape="cic")
        si_units = Units(si_deck)

        lx_internal = si_units.length(si_deck.domain.lx_m)
        ly_internal = si_units.length(si_deck.domain.ly_m)
        amplitude_internal = si_units.e_field(initial_si.amplitude)
        normalized_deck = _fields_deck(FieldsInitial(
            type="seed_em_wave", component="Ez",
            amplitude=amplitude_internal, mode=initial_si.mode))
        normalized_deck.domain = Domain(
            nx=si_deck.domain.nx, ny=si_deck.domain.ny,
            lx_m=lx_internal, ly_m=ly_internal)
        normalized_deck.numerics = Numerics(fdtd_order=4, shape="cic")

        # Both calls receive the solver-internal timestep.  Their E and
        # half-time B seeds must therefore be identical after unit conversion.
        dt_internal = 0.1 * (lx_internal / si_deck.domain.nx)
        si_solver = _RecordingSolver()
        normalized_solver = _RecordingSolver()
        _seed_fields(si_solver, si_deck, dt_internal, si_units)
        _seed_fields(normalized_solver, normalized_deck, dt_internal,
                     Units(normalized_deck))

        si_seed = {component: values for component, values in si_solver.calls}
        normalized_seed = {
            component: values
            for component, values in normalized_solver.calls
        }
        self.assertEqual(si_seed.keys(), normalized_seed.keys())
        for component in si_seed:
            np.testing.assert_allclose(
                si_seed[component], normalized_seed[component],
                rtol=2.0e-15, atol=0.0)

    def test_si_tm_cavity_matches_equivalent_normalized_seed(self):
        walls = BoundaryConfig(
            particle=("specular",) * 4, field=("pec",) * 4)
        initial_si = FieldsInitial(
            type="seed_tm_cavity", component="Ez", amplitude=2.5,
            mode=(1, 2))
        si_deck = _fields_deck(initial_si)
        si_deck.units = "SI"
        si_deck.numerics = Numerics(fdtd_order=4, shape="cic")
        si_deck.boundary = walls
        si_units = Units(si_deck)

        normalized_deck = _fields_deck(FieldsInitial(
            type="seed_tm_cavity", component="Ez",
            amplitude=si_units.e_field(initial_si.amplitude),
            mode=initial_si.mode))
        normalized_deck.domain = Domain(
            nx=si_deck.domain.nx, ny=si_deck.domain.ny,
            lx_m=si_units.length(si_deck.domain.lx_m),
            ly_m=si_units.length(si_deck.domain.ly_m))
        normalized_deck.numerics = Numerics(fdtd_order=4, shape="cic")
        normalized_deck.boundary = walls

        dt_internal = 0.1 * (
            si_units.length(si_deck.domain.lx_m) / si_deck.domain.nx)
        si_solver = _RecordingSolver()
        normalized_solver = _RecordingSolver()
        _seed_fields(si_solver, si_deck, dt_internal, si_units)
        _seed_fields(normalized_solver, normalized_deck, dt_internal,
                     Units(normalized_deck))
        si_seed = dict(si_solver.calls)
        normalized_seed = dict(normalized_solver.calls)
        self.assertEqual(si_seed.keys(), normalized_seed.keys())
        for component in si_seed:
            np.testing.assert_allclose(
                si_seed[component], normalized_seed[component],
                rtol=2.0e-15, atol=0.0)

    def test_seed_perturbation_writes_single_component_sinusoid(self):
        amp, mx = 3.0e-3, 2
        deck = _fields_deck(FieldsInitial(type="seed_perturbation", component="By",
                                          amplitude=amp, mode=(mx, 0)))
        solver = _RecordingSolver()
        _seed_fields(solver, deck)

        # Exactly one divergence-free transverse component is seeded.
        self.assertEqual(len(solver.calls), 1)
        comp, values = solver.calls[0]
        self.assertEqual(comp, "by")

        # By is x-face staggered: all nx+1 physical faces, including the
        # independent high face, are seeded at amp*sin(2*pi*mx*i/nx).
        nx = ny = 8
        g = _core.pic.required_nghost(deck.numerics.fdtd_order)
        pitch, height = nx + 2 * g, ny + 2 * g
        buf = values.reshape(height, pitch)
        i = np.arange(nx + 1)
        expected_row = amp * np.sin(2 * np.pi * mx * i / nx)
        interior = buf[g:g + ny, g:g + nx + 1]
        for row in interior:
            np.testing.assert_allclose(row, expected_row, rtol=0, atol=1e-12)
        # Ghost border is untouched (all zero).
        self.assertEqual(buf[:g].sum(), 0.0)
        self.assertEqual(buf[g + ny:].sum(), 0.0)
        self.assertEqual(buf[:, :g].sum(), 0.0)
        self.assertEqual(buf[:, g + nx + 1:].sum(), 0.0)

    def test_seed_perturbation_rejects_longitudinal_fields(self):
        for component in ("Ex", "Bx"):
            with self.subTest(component=component):
                deck = _fields_deck(FieldsInitial(
                    type="seed_perturbation", component=component,
                    mode=(1, 0)))
                with self.assertRaisesRegex(ValueError, "Gauss|div\\(B\\)"):
                    _seed_fields(_RecordingSolver(), deck)

    def test_si_seed_amplitude_is_converted_by_component_units(self):
        amp_si = 2.5
        deck = _fields_deck(FieldsInitial(
            type="seed_perturbation", component="Ey", amplitude=amp_si,
            mode=(1, 0)))
        deck.units = "SI"
        units = Units(deck)
        solver = _RecordingSolver()
        _seed_fields(solver, deck, units=units)

        _, values = solver.calls[0]
        lattice_peak = float(np.max(np.sin(
            2.0 * np.pi * (np.arange(deck.domain.nx) + 0.5)
            / deck.domain.nx)))
        self.assertAlmostEqual(
            float(np.max(values)),
            lattice_peak * units.e_field(amp_si), places=12)


class BuildParserTests(unittest.TestCase):

    def test_run_defaults(self):
        args = _build_parser().parse_args(["run", "deck.yaml"])
        self.assertEqual(args.input, "deck.yaml")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.log_every, 0)
        self.assertEqual(args.write_every, 0)
        self.assertIsNone(args.steps_override)
        self.assertFalse(args.print_config)

    def test_flag_types(self):
        args = _build_parser().parse_args(
            ["run", "d.yaml", "--seed", "9", "--log-every", "5",
             "--write-every", "20", "--steps-override", "7", "--print-config"])
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.log_every, 5)
        self.assertEqual(args.write_every, 20)
        self.assertEqual(args.steps_override, 7)
        self.assertTrue(args.print_config)

    def test_missing_subcommand_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            _build_parser().parse_args([])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_steps_override_must_be_positive(self):
        for value in ("0", "-3"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as ctx:
                    _build_parser().parse_args(
                        ["run", "deck.yaml", "--steps-override", value])
                self.assertNotEqual(ctx.exception.code, 0)


class FlattenForNpzTests(unittest.TestCase):

    def _final(self):
        return {
            "step": 10,
            "time_s": 1.0e-9,
            "nx": 4,
            "ny": 4,
            "nghost": 1,
            "boundary_field": ("periodic", "periodic",
                               "periodic", "periodic"),
            "external_bx": np.zeros(4),
            "external_by": np.zeros(4),
            "external_bz": np.zeros(4),
            "fields": {"bz": np.arange(16, dtype=float)},
        }

    def test_scalar_series_keys_and_shapes(self):
        series = {"step": [5, 10], "time_s": [0.5e-9, 1.0e-9],
                  "alive_e": [100, 98]}
        flat = _flatten_for_npz([], self._final(), series)
        self.assertIn("series_step", flat)
        self.assertIn("series_alive_e", flat)
        np.testing.assert_array_equal(flat["series_alive_e"], np.array([100, 98]))
        self.assertEqual(flat["series_step"].shape, (2,))

    def test_final_scalars_and_field(self):
        flat = _flatten_for_npz([], self._final(), None)
        self.assertEqual(int(flat["final_step"][0]), 10)
        self.assertEqual(int(flat["nx"][0]), 4)
        self.assertEqual(int(flat["nghost"][0]), 1)
        np.testing.assert_array_equal(
            flat["boundary_field"], np.array(["periodic"] * 4))
        self.assertIn("field_bz", flat)
        self.assertEqual(flat["field_bz"].shape, (16,))

    def test_snapshot_stacking(self):
        snaps = [
            {"step": 5, "time_s": 0.5e-9, "fields": {"bz": np.ones(16)}},
            {"step": 10, "time_s": 1.0e-9, "fields": {"bz": np.zeros(16)}},
        ]
        flat = _flatten_for_npz(snaps, self._final(), None)
        self.assertEqual(flat["snapshot_field_bz"].shape, (2, 16))
        np.testing.assert_array_equal(flat["snapshot_steps"], np.array([5, 10]))


class ExactEndTimeTests(unittest.TestCase):

    class _Solver:
        def __init__(self):
            self.steps = []
            self.finalized = False

        def step(self, dt):
            self.steps.append(float(dt))

        def finalize(self):
            self.finalized = True

        def field_component_to_host(self, _component):
            return np.zeros(4)

        def external_field_component_to_host(self, _component):
            return np.zeros(4)

    def test_t_end_clips_last_step_and_output_time_exactly(self):
        deck = PicDeck(
            domain=Domain(nx=2, ny=2, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[],
            time=Time(dt_s=0.4, steps=10, t_end_s=1.0),
            diagnostics=Diagnostics(fields=[], per_species=False),
            units="normalized")
        solver = self._Solver()
        args = SimpleNamespace(log_every=0, write_every=0)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.npz"
            _run_loop(solver, deck, [], Units(deck), 0.4, 0.4,
                      out_path, args)
            out = np.load(out_path, allow_pickle=False)
            self.assertEqual(float(out["final_time_s"][0]), 1.0)
            self.assertEqual(int(out["final_step"][0]), 3)
        np.testing.assert_allclose(solver.steps, [0.4, 0.4, 0.2],
                                   rtol=0.0, atol=1.0e-15)
        self.assertTrue(solver.finalized)

    def test_first_clipped_step_uses_matching_field_half_step_seed(self):
        class _SeededSolver(self._Solver):
            def __init__(self):
                super().__init__()
                self.seed_calls = []

            def seed_field(self, component, values):
                self.seed_calls.append((component, np.asarray(values)))

        deck = PicDeck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[],
            fields=Fields(initial=FieldsInitial(
                type="seed_em_wave", component="Ez", mode=(1, 0))),
            time=Time(dt_s=0.08, steps=10, t_end_s=0.02),
            diagnostics=Diagnostics(fields=[], per_species=False),
            units="normalized")
        deck.validate()
        units = Units(deck)
        solver = _SeededSolver()
        with patch("quasar.pic.cli._make_solver", return_value=solver):
            got_solver, indices, dt, dt_si = prepare_run(deck, units)
        self.assertIs(got_solver, solver)
        self.assertEqual(indices, [])
        self.assertEqual(dt, 0.08)  # nominal cadence remains unchanged

        expected = _RecordingSolver()
        _seed_fields(expected, deck, dt=0.02, units=units)
        self.assertEqual(
            [name for name, _ in solver.seed_calls],
            [name for name, _ in expected.calls])
        for (_, actual), (_, reference) in zip(solver.seed_calls, expected.calls):
            np.testing.assert_allclose(actual, reference, rtol=0.0, atol=0.0)

        args = SimpleNamespace(log_every=0, write_every=0)
        with tempfile.TemporaryDirectory() as tmp:
            _run_loop(solver, deck, [], units, dt, dt_si,
                      Path(tmp) / "out.npz", args)
        self.assertEqual(solver.steps, [0.02])
        self.assertTrue(solver.finalized)

    def test_t_end_clips_in_solver_units_before_assigning_si_label(self):
        class _ScaledUnits:
            factor = np.pi

            def time(self, value):
                return value * self.factor

            def time_to_si(self, value):
                return value / self.factor

            @staticmethod
            def field_component_to_si(_name, value):
                return value

        deck = PicDeck(
            domain=Domain(nx=2, ny=2, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[],
            time=Time(dt_s=0.1, steps=10, t_end_s=0.137),
            diagnostics=Diagnostics(fields=[], per_species=False),
            units="normalized")
        units = _ScaledUnits()
        solver = self._Solver()
        args = SimpleNamespace(log_every=0, write_every=0)
        dt_si = 0.1
        dt = units.time(dt_si)
        expected_last = units.time(deck.time.t_end_s) - dt
        # Converting the rounded SI remainder back is observably different; this
        # is the mismatch the regression guards against.
        self.assertNotEqual(expected_last,
                            units.time(deck.time.t_end_s - dt_si))
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.npz"
            _run_loop(solver, deck, [], units, dt, dt_si, out_path, args)
            out = np.load(out_path, allow_pickle=False)
            self.assertEqual(float(out["final_time_s"][0]),
                             deck.time.t_end_s)
        self.assertEqual(solver.steps, [dt, expected_last])


class _ComponentRecordingSolver:
    def __init__(self):
        self.field_calls = []
        self.external_calls = []

    def field_component_to_host(self, component):
        self.field_calls.append(component)
        return np.full(4, len(self.field_calls), dtype=float)

    def external_field_component_to_host(self, component):
        self.external_calls.append(component)
        return np.full(4, len(self.external_calls), dtype=float)


class SnapshotTests(unittest.TestCase):

    def test_snapshot_reads_only_requested_field_components(self):
        deck = PicDeck(
            domain=Domain(nx=2, ny=2, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            diagnostics=Diagnostics(fields=["bz"], per_species=False),
            units="normalized",
        )
        solver = _ComponentRecordingSolver()
        snap = _snapshot(solver, deck, [], step=3, sim_time=1.5, units=Units(deck))

        self.assertEqual(solver.field_calls, ["bz"])
        self.assertEqual(solver.external_calls, ["bx", "by", "bz"])
        self.assertEqual(snap["nx"], 2)
        self.assertEqual(snap["ny"], 2)
        self.assertEqual(snap["boundary_field"], tuple(deck.boundary.field))
        self.assertEqual(set(snap["fields"]), {"bz"})

    def test_snapshot_applies_si_conversion_per_component(self):
        # An SI deck has non-unit field scales, so the snapshot must multiply each
        # raw component by the matching E- or B-field scale (ez uses E, bz uses B).
        deck = PicDeck(
            domain=Domain(nx=2, ny=2, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            diagnostics=Diagnostics(fields=["ez", "bz"], per_species=False),
            units="SI",
        )
        units = Units(deck)
        solver = _ComponentRecordingSolver()
        snap = _snapshot(solver, deck, [], step=0, sim_time=0.0, units=units)

        # field_calls records ez then bz -> raw arrays full of 1 then 2.
        np.testing.assert_allclose(
            snap["fields"]["ez"], units.field_component_to_si("ez", np.full(4, 1.0)))
        np.testing.assert_allclose(
            snap["fields"]["bz"], units.field_component_to_si("bz", np.full(4, 2.0)))
        # external_calls records bx, by, bz -> raw 1, 2, 3.
        np.testing.assert_allclose(
            snap["external_bz"], units.field_component_to_si("bz", np.full(4, 3.0)))


class RealBindingAccessorTests(unittest.TestCase):
    """Exercise the C++ field_component_to_host accessor on a real solver so a
    binding/layout regression or the unknown-component error path is caught."""

    def _solver(self):
        deck = PicDeck(
            domain=Domain(nx=4, ny=4, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                             initial=SpeciesInitial())],
            units="normalized",
        )
        return _make_solver(deck, Units(deck)), deck

    def test_component_to_host_matches_storage_size(self):
        solver, _ = self._solver()
        arr = solver.field_component_to_host("bz")
        self.assertEqual(arr.shape, (solver.storage_size(),))

    def test_component_matches_whole_dict_fetch(self):
        solver, _ = self._solver()
        np.testing.assert_array_equal(
            solver.field_component_to_host("ez"), solver.fields_to_host()["ez"])
        np.testing.assert_array_equal(
            solver.external_field_component_to_host("bx"),
            solver.external_fields_to_host()["bx"])

    def test_unknown_component_raises(self):
        solver, _ = self._solver()
        with self.assertRaises(ValueError):
            solver.field_component_to_host("pressure")
        with self.assertRaises(ValueError):
            solver.external_field_component_to_host("nope")


class _RecordingSpecies:
    """Captures set_host_particles(...) kwargs without a GPU."""

    def __init__(self):
        self.host = None

    def set_host_particles(self, **kwargs):
        self.host = {k: np.asarray(v) for k, v in kwargs.items()}


class _SpeciesRecordingSolver:
    """Stub solver capturing what _seed_species hands the sampler.

    The sampling arithmetic lives in kernels now, so this records the
    ParticleSampleConfig the CLI assembled -- which is the part under test --
    and then runs the real sampler on it to get the arrays. The one thing it
    does not do is override the domain from a solver grid, since there is no
    grid here; the CLI has already filled the domain fields from the deck.
    """

    def __init__(self):
        self.configs = []
        self.sample_configs = []
        self.species = []

    def add_species(self, cfg):
        self.configs.append(cfg)
        self.species.append(_RecordingSpecies())
        return len(self.species) - 1

    def species_at(self, idx):
        return self.species[idx]

    def sample_species_particles(self, idx, config):
        self.sample_configs.append(config)
        sample = _core.pic.sample_particles(config)
        self.species[idx].host = {
            key: np.asarray(value) for key, value in sample.items()}


def _species_deck(initial: SpeciesInitial) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=64,
                         initial=initial)],
        units="normalized",
    )


class SeedSpeciesTests(unittest.TestCase):
    """The maxwellian_block branch is only reachable on a GPU end-to-end run, so
    pin its CPU-computable contract (positions confined to the region; macro
    weight derived from the block area, not the full domain) with a stub solver."""

    def test_block_positions_confined_to_region_and_weight_uses_block_area(self):
        region = dict(region_x_min_m=0.25, region_x_max_m=0.75,
                      region_y_min_m=0.5, region_y_max_m=1.0)
        deck = _species_deck(SpeciesInitial(distribution="maxwellian_block",
                                            density_per_m3=1.0e18,
                                            temperature_eV=1.0e-4, **region))
        units = Units(deck)
        solver = _SpeciesRecordingSolver()
        rng = np.random.default_rng(0)
        _seed_species(solver, deck, units, 0)

        host = solver.species[0].host
        self.assertIsNotNone(host)
        x_lo = units.length(region["region_x_min_m"])
        x_hi = units.length(region["region_x_max_m"])
        y_lo = units.length(region["region_y_min_m"])
        y_hi = units.length(region["region_y_max_m"])
        self.assertTrue(np.all(host["x"] >= x_lo - 1e-12))
        self.assertTrue(np.all(host["x"] <= x_hi + 1e-12))
        self.assertTrue(np.all(host["y"] >= y_lo - 1e-12))
        self.assertTrue(np.all(host["y"] <= y_hi + 1e-12))
        # All positions lie inside the block, never the full domain.
        self.assertLess(host["x"].max(), units.length(deck.domain.lx_m))

        # Macro weight = density * block_area / N (block area, not domain area).
        block_area = (x_hi - x_lo) * (y_hi - y_lo)
        expected_w = units.density(1.0e18) * block_area / 64
        np.testing.assert_allclose(host["weight"], expected_w, rtol=1e-12)

    def test_uniform_positions_span_full_domain(self):
        deck = _species_deck(SpeciesInitial(distribution="maxwellian_uniform",
                                            density_per_m3=1.0e18,
                                            temperature_eV=1.0e-4))
        units = Units(deck)
        solver = _SpeciesRecordingSolver()
        _seed_species(solver, deck, units, 0)
        host = solver.species[0].host
        # Uniform weight uses the full domain area, so it differs from a sub-block.
        domain_area = units.length(deck.domain.lx_m) * units.length(deck.domain.ly_m)
        expected_w = units.density(1.0e18) * domain_area / 64
        np.testing.assert_allclose(host["weight"], expected_w, rtol=1e-12)

    def test_velocity_perturbation_seeds_the_requested_resolved_mode(self):
        perturbation = VelocityPerturbation(
            amplitude_v=(1.0e-3, -2.0e-3, 0.0), mode=(1, 0),
            phase_rad=0.25)
        deck = _species_deck(SpeciesInitial(
            distribution="maxwellian_uniform", density_per_m3=1.0,
            temperature_eV=0.0, velocity_perturbation=perturbation))
        deck.validate()
        units = Units(deck)
        solver = _SpeciesRecordingSolver()
        _seed_species(solver, deck, units, 0)
        host = solver.species[0].host
        phase = 2.0 * np.pi * host["x"] + perturbation.phase_rad
        np.testing.assert_allclose(
            host["vx"], perturbation.amplitude_v[0] * np.sin(phase),
            rtol=0.0, atol=2.0e-18)
        np.testing.assert_allclose(
            host["vy"], perturbation.amplitude_v[1] * np.sin(phase),
            rtol=0.0, atol=2.0e-18)
        np.testing.assert_array_equal(host["vz"], 0.0)

    def test_macro_weight_rejects_true_underflow_but_not_false_intermediate_range(self):
        self.assertAlmostEqual(
            _macro_weight(1.0, 1.0e-300, 1.0e300, "e"), 1.0)
        with self.assertRaisesRegex(OverflowError, "macro-particle weight"):
            _macro_weight(
                1.0, np.nextafter(0.0, 1.0), 0.5, "underflow")


class _ExternalRecordingSolver:
    """Captures the evaluator + scales passed to sample_external_field."""

    def __init__(self):
        self.evaluator = None
        self.conductors = None
        self.scales = None
        self.plane = None

    def sample_external_field(self, evaluator, conductors, length_scale,
                              e_field_scale, b_field_scale, plane="xy"):
        self.evaluator = evaluator
        self.conductors = conductors
        self.scales = (length_scale, e_field_scale, b_field_scale)
        self.plane = plane


def _external_deck(ef: ExternalField) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        external_field=ef,
        units="SI",
    )


class ApplyExternalFieldTests(unittest.TestCase):
    """_apply_external_field builds the evaluator by registry name then pushes deck
    params through configure(); this is only reached end-to-end on a GPU run, so
    exercise the (CPU-only) build+configure seam here against the real _core."""

    def _eval_for(self, ef: ExternalField):
        deck = _external_deck(ef)
        solver = _ExternalRecordingSolver()
        _apply_external_field(solver, deck, Units(deck))
        self.assertIsNotNone(solver.evaluator)
        return solver.evaluator

    def test_uniform_builds_uniform_evaluator_and_configures_b(self):
        ev = self._eval_for(ExternalField(evaluator_type="uniform",
                                          uniform_b=(0.0, 0.0, 2.0)))
        self.assertEqual(type(ev).__name__, "UniformEvaluator")
        # configure() must have applied the deck B0: evaluate_B returns it.
        cs = _core.magnetostatics.ConductorSystem()
        pts = _core.magnetostatics.PointCloud()
        pts.add(_core.Vec3(0.1, 0.2, 0.3))
        b = ev.evaluate_B(cs, pts)
        np.testing.assert_allclose(b[0], [0.0, 0.0, 2.0], atol=1e-12)

    def test_dipole_builds_dipole_evaluator(self):
        ev = self._eval_for(ExternalField(evaluator_type="dipole",
                                          dipole_moment=(0.0, 0.0, 1.0)))
        self.assertEqual(type(ev).__name__, "DipoleEvaluator")

    def test_gradient_builds_gradient_evaluator(self):
        grad = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0))
        ev = self._eval_for(ExternalField(evaluator_type="gradient",
                                          gradient_b0=(0.0, 0.0, 1.0),
                                          gradient_matrix=grad))
        self.assertEqual(type(ev).__name__, "GradientEvaluator")

    def test_biot_savart_builds_biot_savart_evaluator(self):
        ev = self._eval_for(ExternalField(evaluator_type="biot_savart",
                                          conductors=[]))
        self.assertEqual(type(ev).__name__, "BiotSavartEvaluator")

    def test_plane_passed_through_default_xy(self):
        deck = _external_deck(ExternalField(evaluator_type="uniform",
                                            uniform_b=(0.0, 0.0, 1.0)))
        solver = _ExternalRecordingSolver()
        _apply_external_field(solver, deck, Units(deck))
        self.assertEqual(solver.plane, "xy")

    def test_plane_xz_passed_through(self):
        deck = _external_deck(ExternalField(evaluator_type="uniform",
                                            uniform_b=(0.0, 0.0, 1.0)))
        deck.plane = "xz"
        solver = _ExternalRecordingSolver()
        _apply_external_field(solver, deck, Units(deck))
        self.assertEqual(solver.plane, "xz")


class CartesianCflGuardTests(unittest.TestCase):
    """The Cartesian explicit-dt over-CFL rejection in prepare_run mirrors the
    cylindrical guard (tested in test_pic_io_cylindrical.py). Both branches are
    near-identical, but only the cylindrical one had coverage; pin the Cartesian
    (the common) path too. These build the C++ solver, so skip without a device."""

    def _cartesian_deck(self, dt_s):
        return PicDeck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[Species(name="e", charge_C=-1.0, mass_kg=1.0,
                             n_particles=64,
                             initial=SpeciesInitial(temperature_eV=1.0e-4))],
            time=Time(dt_s=dt_s, steps=4),
            units="normalized",
        )

    def test_explicit_dt_above_cartesian_cfl_rejected(self):
        from quasar.pic.cli import prepare_run
        deck = self._cartesian_deck(1.0)  # 1 s vs a sub-second internal limit
        deck.validate()
        try:
            with self.assertRaisesRegex(ValueError, r"CFL stability limit"):
                prepare_run(deck, Units(deck))
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")

    def test_auto_dt_within_cartesian_cfl_limit(self):
        from quasar.pic.cli import _cfl_limit_internal, prepare_run
        deck = self._cartesian_deck("auto")
        deck.validate()
        try:
            _solver, _idx, dt, _dt_si = prepare_run(deck, Units(deck))
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")
        self.assertLessEqual(dt, _cfl_limit_internal(deck.domain, Units(deck), 2))


if __name__ == "__main__":
    unittest.main()
