"""Tests for the standalone PX4 MuJoCo wind simulation."""

from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib import colors as matplotlib_colors


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/wind_field_simulation/simulation.py"
)


def test_simulation_script_is_in_standalone_package() -> None:
    """The extracted application resides inside its own package."""
    assert SCRIPT_PATH.is_file()
    assert SCRIPT_PATH.parent.name == "wind_field_simulation"


def load_simulation_module():
    """Load the package once without opening sockets or graphical windows."""
    from wind_field_simulation import simulation

    return simulation


def test_default_resources_are_bundled() -> None:
    """Default paths use package resources rather than sibling repositories."""
    simulation = load_simulation_module()
    resource_root = SCRIPT_PATH.parent / "assets"
    assert simulation.default_pinn_project_root() == resource_root
    assert simulation.default_pinn_checkpoint_dir() == resource_root / "checkpoints"
    assert simulation.default_xml_path().is_relative_to(resource_root)
    assert simulation.default_params_path().is_relative_to(resource_root)


def test_pinn_feature_builder_matches_checkpoint_contract() -> None:
    """Verifies the checkpoint feature order, frames, and motor values."""
    simulation = load_simulation_module()

    features, relative_velocity = (
        simulation.PinnWindResidualModel.build_features(
            velocity_world=np.array([3.0, -2.0, 1.0]),
            quaternion_wxyz=np.array([0.9, 0.1, 0.2, 0.3]),
            angular_velocity_body=np.array([0.4, -0.5, 0.6]),
            wind_world=np.array([1.0, 2.0, -1.0]),
        )
    )

    assert features.dtype == np.float32
    assert features.shape == (17,)
    np.testing.assert_allclose(relative_velocity, [2.0, -4.0, 2.0])
    np.testing.assert_allclose(features[0:3], relative_velocity)
    np.testing.assert_allclose(features[3:7], [0.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(features[7:10], [0.4, -0.5, 0.6])
    np.testing.assert_allclose(
        features[10:14],
        np.full(4, simulation.HOVER_OMEGA_RAD_S),
    )
    np.testing.assert_allclose(features[14:17], [1.0, 2.0, -1.0])


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("velocity_world", np.zeros(2)),
        ("quaternion_wxyz", np.zeros(3)),
        ("angular_velocity_body", np.array([0.0, np.nan, 0.0])),
        ("wind_world", np.zeros(4)),
    ],
)
def test_pinn_feature_builder_rejects_invalid_state(
    field_name: str,
    value: np.ndarray,
) -> None:
    """Rejects malformed or non-finite PINN state components."""
    simulation = load_simulation_module()
    arguments = {
        "velocity_world": np.zeros(3),
        "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "angular_velocity_body": np.zeros(3),
        "wind_world": np.zeros(3),
    }
    arguments[field_name] = value

    with pytest.raises(ValueError):
        simulation.PinnWindResidualModel.build_features(**arguments)


def test_pinn_predictor_matches_reference_checkpoint_on_cpu() -> None:
    """Matches the existing checkpoint's fixed-state reference prediction."""
    simulation = load_simulation_module()
    predictor = simulation.PinnWindResidualModel(
        device=simulation.torch.device("cpu"),
    )

    prediction = predictor.predict(
        velocity_world=np.zeros(3),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_body=np.zeros(3),
        wind_world=np.array([8.0, 0.0, 0.0]),
    )

    assert prediction.shape == (3,)
    assert prediction.dtype == np.float64
    np.testing.assert_allclose(
        prediction,
        [7.2798495, -0.00036034, 0.03778124],
        rtol=2e-5,
        atol=2e-5,
    )


def test_pinn_predictor_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """Reports the exact missing model file before inference starts."""
    simulation = load_simulation_module()

    with pytest.raises(FileNotFoundError, match="best_model.pt"):
        simulation.PinnWindResidualModel(
            checkpoint_dir=tmp_path,
            device=simulation.torch.device("cpu"),
        )


def test_residual_filter_applies_ema_and_all_three_axes() -> None:
    """Applies the approved EMA without masking the vertical component."""
    simulation = load_simulation_module()
    filter_state = simulation.ResidualSafetyFilter()

    first = filter_state.update(np.array([4.0, -2.0, 1.0]))
    second = filter_state.update(np.array([4.0, -2.0, 1.0]))

    np.testing.assert_allclose(first, [0.2, -0.1, 0.05])
    np.testing.assert_allclose(second, [0.39, -0.195, 0.0975])


def test_residual_filter_limits_and_resets() -> None:
    """Limits every axis and clears the stored EMA state."""
    simulation = load_simulation_module()
    filter_state = simulation.ResidualSafetyFilter(ema_alpha=1.0)

    limited = filter_state.update(np.array([10.0, -7.0, 8.0]))
    np.testing.assert_allclose(limited, [6.0, -6.0, 6.0])

    filter_state.reset()
    np.testing.assert_allclose(filter_state.filtered, np.zeros(3))


def test_constant_wind_returns_copy_or_zero() -> None:
    simulation = load_simulation_module()
    wind = simulation.ConstantWindField(np.array([2.0, -2.0, 0.0]))

    active = wind.velocity(enabled=True)
    active[0] = 100.0

    np.testing.assert_allclose(wind.velocity(enabled=True), [2.0, -2.0, 0.0])
    np.testing.assert_allclose(wind.velocity(enabled=False), np.zeros(3))


def test_panel_wind_fields_show_initial_configuration() -> None:
    """Display explicit CLI drafts without permitting wind activation."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(
        queue.SimpleQueue(), initial_wind_world=(-2.0, 2.0, 0.0),
    )
    try:
        assert [float(box.text) for box in panel.wind_inputs] == [-2.0, 2.0]
        assert not panel.buttons["wind"].get_active()
    finally:
        panel.close()


def test_panel_only_submits_horizontal_wind_components() -> None:
    """Expose X/Y only and keep the internal PINN wind vector's Z zero."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        assert [box.label.get_text() for box in panel.wind_inputs] == ["X", "Y"]
        for box, value in zip(panel.wind_inputs, ("-2.5", "1")):
            box.set_val(value)
        panel._on_apply_wind(None)
        assert commands.get_nowait() == simulation.UiCommand(
            simulation.UiCommandType.SET_WIND, (-2.5, 1.0, 0.0),
        )
    finally:
        panel.close()


@pytest.mark.parametrize("vertical", [-1.0, 0.5])
def test_vertical_wind_commands_preserve_horizontal_configuration(
    vertical: float,
) -> None:
    """Reject vertical wind before setup and preserve accepted horizontal wind."""
    simulation = load_simulation_module()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=RecordingPinnPredictor(),
    )
    invalid = simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (2.0, 3.0, vertical),
    )
    app._process_ui_command(invalid)
    assert app.wind_field is None
    assert not app.flight_control.wind_configured
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, -2.0, 0.0),
    ))
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
    app._process_ui_command(invalid)
    assert app.flight_control.wind_enabled
    np.testing.assert_allclose(app.wind_field.velocity(True), [1.0, -2.0, 0.0])


def test_cli_wind_draft_is_horizontal_only() -> None:
    """Only horizontal CLI draft flags remain, with Z fixed to zero."""
    simulation = load_simulation_module()
    draft = simulation.parse_config(["--wind-x", "1", "--wind-y", "-2"])
    np.testing.assert_allclose(draft.wind_velocity_world, [1.0, -2.0, 0.0])
    with pytest.raises(SystemExit):
        simulation.parse_config(["--wind-z", "0"])


def test_default_wind_is_absent_and_panel_switch_is_disabled() -> None:
    """Startup must not invent parameters or allow direct toggle callbacks."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        assert simulation.parse_config([]).wind_velocity_world is None
        assert [box.text for box in panel.wind_inputs] == ["", ""]
        assert not panel.buttons["wind"].get_active()
        panel._on_toggle_wind(None)
        panel._on_apply_wind(None)
        assert commands.empty()
        assert not panel.buttons["wind"].get_active()
    finally:
        panel.close()


@pytest.mark.parametrize("cli", [[], ["--wind-x", "3"]])
def test_unconfigured_runtime_blocks_wind_even_with_cli_draft(cli: list[str]) -> None:
    """No accepted field exists until Apply, including with CLI prefills."""
    simulation = load_simulation_module()
    predictor = RecordingPinnPredictor()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer", *cli]),
        bridge=RecordingBridge(), pinn_predictor=predictor,
    )
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
    assert not app.flight_control.wind_enabled
    assert app.wind_field is None
    assert app._panel_snapshot(app._current_state()[0]).configured_wind_world is None
    active, _relative, force = app._apply_wind_force()
    np.testing.assert_allclose(active, np.zeros(3))
    np.testing.assert_allclose(force, np.zeros(3))
    assert predictor.calls == []
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (float("nan"), 0.0, 0.0),
    ))
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
    assert not app.flight_control.wind_enabled
    assert app.wind_field is None


@pytest.mark.parametrize("velocity", [(1.0, -2.0, 0.0), (0.0, 0.0, 0.0)])
def test_apply_acknowledgement_unlocks_wind_switch(
    velocity: tuple[float, float, float],
) -> None:
    """Only accepted settings unlock a real switch click, including zero wind."""
    simulation = load_simulation_module()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=RecordingPinnPredictor(),
    )
    panel = simulation.ControlPanel(app.command_queue)
    try:
        panel.update(app._panel_snapshot(app._current_state()[0]))
        for box, value in zip(panel.wind_inputs, velocity):
            box.set_val(str(value))
        panel._on_apply_wind(None)
        assert not panel.buttons["wind"].get_active()
        app._drain_ui_commands()
        assert not app.flight_control.wind_enabled
        panel.update(app._panel_snapshot(app._current_state()[0]))
        assert panel.buttons["wind"].get_active()

        panel.figure.canvas.draw()
        pixel_x, pixel_y = panel.buttons["wind"].ax.transAxes.transform((0.5, 0.5))
        for name in ("button_press_event", "button_release_event"):
            panel.figure.canvas.callbacks.process(name, MouseEvent(
                name, panel.figure.canvas, pixel_x, pixel_y, button=1,
            ))
        app._drain_ui_commands()
        assert app.flight_control.wind_enabled
        panel._on_toggle_wind(None)
        app._drain_ui_commands()
        assert not app.flight_control.wind_enabled
        assert app.wind_field is not None
    finally:
        panel.close()


def test_panel_apply_wind_enqueues_one_complete_vector() -> None:
    """Editing is inert until Apply submits all three values atomically."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        for box, value in zip(panel.wind_inputs, ("-2.5", "1.25")):
            box.set_val(value)
        assert commands.empty()

        panel._on_apply_wind(None)

        assert commands.get_nowait() == simulation.UiCommand(
            simulation.UiCommandType.SET_WIND, (-2.5, 1.25, 0.0),
        )
        assert commands.empty()
    finally:
        panel.close()


def test_wind_apply_button_click_reaches_command_queue() -> None:
    """Route an actual mouse click through the button without position jogs."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        for box, value in zip(panel.wind_inputs, ("-3", "0")):
            box.set_val(value)
        panel.figure.canvas.draw()
        pixel_x, pixel_y = panel.buttons["apply_wind"].ax.transAxes.transform(
            (0.5, 0.5),
        )
        for name in ("button_press_event", "button_release_event"):
            event = MouseEvent(
                name, panel.figure.canvas, pixel_x, pixel_y, button=1,
            )
            panel.figure.canvas.callbacks.process(name, event)
        assert commands.get_nowait() == simulation.UiCommand(
            simulation.UiCommandType.SET_WIND, (-3.0, 0.0, 0.0),
        )
        assert commands.empty()
    finally:
        panel.close()


@pytest.mark.parametrize("invalid", ["", "oops", "nan", "inf", "-inf", "1e309"])
def test_panel_invalid_wind_preserves_settings_and_error(invalid: str) -> None:
    """Reject the whole edit and retain visible feedback across refreshes."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        panel.wind_inputs[0].set_val("2")
        panel.wind_inputs[1].set_val(invalid)
        panel._on_apply_wind(None)
        panel.update(make_panel_snapshot(
            simulation, configured_wind_world=None,
        ))

        assert commands.empty()
        assert not panel.buttons["wind"].get_active()
        assert "finite" in panel.wind_feedback_text.get_text().lower()
        assert panel.wind_inputs[1].text == invalid
    finally:
        panel.close()


def test_panel_refresh_preserves_new_draft_after_prior_submission() -> None:
    """An acknowledgement must not overwrite a newer in-progress edit."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())
    try:
        panel.update(make_panel_snapshot(
            simulation, configured_wind_world=np.array([1.0, 0.0, 0.0]),
        ))
        panel.wind_inputs[0].set_val("2")
        panel._on_apply_wind(None)
        panel.wind_inputs[0].set_val("-")

        panel.update(make_panel_snapshot(
            simulation, configured_wind_world=np.array([2.0, 0.0, 0.0]),
        ))

        assert panel.wind_inputs[0].text == "-"
        assert "2" in panel.wind_feedback_text.get_text()
    finally:
        panel.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_wind_edit_reaches_force_and_snapshot(enabled: bool) -> None:
    """Apply queued settings without changing wind enable, CLI or flight state."""
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    predictor = RecordingPinnPredictor()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer", "--wind-x", "3"]),
        bridge=bridge, pinn_predictor=predictor,
    )
    app.flight_control.wind_enabled = enabled
    app.residual_filter.update([1.0, 2.0, 3.0])
    filter_before = app.residual_filter.filtered.copy()
    target_before = app.flight_control.target_position_world.copy()
    app.command_queue.put(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (-2.0, 0.5, 0.0),
    ))
    app._drain_ui_commands()

    assert app.flight_control.wind_enabled is enabled
    assert app.flight_control.flight_state is simulation.FlightState.DISARMED
    np.testing.assert_allclose(app.residual_filter.filtered, filter_before)
    np.testing.assert_allclose(app.flight_control.target_position_world, target_before)
    np.testing.assert_allclose(app.config.wind_velocity_world, [3.0, 0.0, 0.0])
    active, _relative, _force = app._apply_wind_force()
    snapshot = app._panel_snapshot(app._current_state()[0])
    np.testing.assert_allclose(snapshot.configured_wind_world, [-2.0, 0.5, 0.0])
    np.testing.assert_allclose(
        active, [-2.0, 0.5, 0.0] if enabled else [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(snapshot.active_wind_world, active)
    if enabled:
        np.testing.assert_allclose(
            predictor.calls[-1]["wind_world"], [-2.0, 0.5, 0.0],
        )
    else:
        assert predictor.calls == []
    snapshot.configured_wind_world[:] = 99.0
    np.testing.assert_allclose(app.wind_field.velocity(True), [-2.0, 0.5, 0.0])
    assert bridge.targets == []


@pytest.mark.parametrize("payload", [(1.0, float("nan"), 3.0), (1.0, 2.0), None])
def test_runtime_rejects_invalid_wind_command(payload: Any) -> None:
    """Malformed queued commands cannot partially update wind or crash flight."""
    simulation = load_simulation_module()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=RecordingPinnPredictor(),
    )
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, payload,
    ))
    np.testing.assert_allclose(app.wind_field.velocity(True), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("unsafe", ["1e20", "1e40"])
def test_panel_rejects_wind_that_overflows_pinn_arithmetic(unsafe: str) -> None:
    """Reject finite float64 values unsafe for float32 norm and drag."""
    simulation = load_simulation_module()
    commands: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(commands)
    try:
        for box, value in zip(panel.wind_inputs, (unsafe, "0")):
            box.set_val(value)
        panel._on_apply_wind(None)
        assert commands.empty()
        assert "numeric range" in panel.wind_feedback_text.get_text()
    finally:
        panel.close()


@pytest.mark.parametrize("unsafe", [1e20, 1e40])
def test_unsafe_wind_edit_preserves_real_pinn_inference(unsafe: float) -> None:
    """A rejected runtime update must leave real CPU inference operational."""
    simulation = load_simulation_module()
    predictor = simulation.PinnWindResidualModel(
        device=simulation.torch.device("cpu"),
    )
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=predictor,
    )
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app.flight_control.wind_enabled = True
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (unsafe, 0.0, 0.0),
    ))
    active, _relative, force = app._apply_wind_force()
    np.testing.assert_allclose(active, [1.0, 0.0, 0.0])
    assert np.isfinite(force).all()


class FakeUserScene:
    """Minimal writable MuJoCo user-scene test double."""

    def __init__(self, maxgeom: int = 300, ngeom: int = 0) -> None:
        self.ngeom = ngeom
        self.maxgeom = maxgeom
        self.geoms = [object() for _ in range(maxgeom)]


def install_scene_geometry_recorders(
    simulation: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple], list[dict[str, object]]]:
    """Records initialized geometries and capsule endpoints."""
    init_calls: list[tuple] = []
    connector_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        simulation.mujoco,
        "mjv_initGeom",
        lambda *args: init_calls.append(args),
    )

    def record_connector(geom, geom_type, width, start, end) -> None:
        del geom, geom_type
        connector_calls.append(
            {
                "width": width,
                "start": np.asarray(start).copy(),
                "end": np.asarray(end).copy(),
            }
        )

    monkeypatch.setattr(
        simulation.mujoco,
        "mjv_connector",
        record_connector,
    )
    return init_calls, connector_calls


def test_zero_wind_clears_scene_and_draws_no_arrows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    scene = FakeUserScene(ngeom=9)
    init_calls, _connector_calls = install_scene_geometry_recorders(
        simulation,
        monkeypatch,
    )

    simulation.draw_wind_arrows(scene, np.zeros(3), 1.0)

    assert scene.ngeom == 0
    assert init_calls == []


def test_nonzero_wind_draws_reference_arrow_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    scene = FakeUserScene()
    init_calls, connector_calls = install_scene_geometry_recorders(
        simulation,
        monkeypatch,
    )

    simulation.draw_wind_arrows(scene, [8.0, 0.0, 0.0], 0.0)

    assert scene.ngeom == 240
    assert len(init_calls) == 240
    assert len(connector_calls) == 120
    assert sum(
        call[1] == simulation.mujoco.mjtGeom.mjGEOM_TRIANGLE
        for call in init_calls
    ) == 120
    assert not any(
        call[1] == simulation.mujoco.mjtGeom.mjGEOM_SPHERE
        for call in init_calls
    )
    first_arrow = connector_calls[0]
    np.testing.assert_allclose(first_arrow["start"], [-10.0, -8.0, 0.0])
    displacement = first_arrow["end"] - first_arrow["start"]
    assert displacement[0] == pytest.approx(0.245, rel=1e-5)
    np.testing.assert_allclose(displacement[1:], np.zeros(2), atol=1e-12)
    assert first_arrow["width"] == pytest.approx(0.009)


def test_wind_arrow_grid_uses_simulation_time_flow_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    scene = FakeUserScene()
    _init_calls, connector_calls = install_scene_geometry_recorders(
        simulation,
        monkeypatch,
    )

    simulation.draw_wind_arrows(scene, [8.0, 0.0, 0.0], 0.125)

    np.testing.assert_allclose(
        connector_calls[0]["start"],
        [-9.0, -8.0, 0.0],
    )


def test_wind_arrow_drawing_stops_at_scene_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    scene = FakeUserScene(maxgeom=3)
    init_calls, _connector_calls = install_scene_geometry_recorders(
        simulation,
        monkeypatch,
    )

    simulation.draw_wind_arrows(scene, [8.0, 0.0, 0.0], 0.0)

    assert scene.ngeom == 2
    assert len(init_calls) == 2


@pytest.mark.parametrize(
    ("wind", "expected_direction"),
    [([1, 0, 9], [1, 0, 0]), ([0, -2, -7], [0, -1, 0]),
     ([1, 1, 0], [2**-0.5, 2**-0.5, 0])],
)
def test_wind_triangle_and_shaft_stay_in_world_horizontal_plane(
    wind: list[float], expected_direction: list[float],
) -> None:
    """Check actual MuJoCo geoms, including all three triangle vertices."""
    simulation = load_simulation_module()
    model = simulation.mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = simulation.mujoco.MjvScene(model, maxgeom=300)
    original = np.array(wind, dtype=float)
    simulation.draw_wind_arrows(scene, original, 0)
    np.testing.assert_array_equal(original, wind)
    assert scene.ngeom == 240
    for index in range(0, scene.ngeom, 2):
        shaft, head = scene.geoms[index:index + 2]
        assert head.type == simulation.mujoco.mjtGeom.mjGEOM_TRIANGLE
        rotation = head.mat.reshape(3, 3)
        local_vertices = np.array([
            [0, 0, 0], [head.size[0], 0, 0], [0, head.size[1], 0],
        ])
        vertices = local_vertices @ rotation.T + head.pos
        np.testing.assert_allclose(vertices[:, 2], shaft.pos[2], atol=1e-7)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(
            shaft.mat.reshape(3, 3)[:, 2], expected_direction, atol=1e-7,
        )
        # The triangle tip is ahead of the symmetric rear edge along the wind.
        base_center = 0.5 * (vertices[1] + vertices[2])
        np.testing.assert_allclose(
            vertices[0] - base_center,
            np.array(expected_direction) * 0.035, atol=1e-6,
        )


def test_wind_grid_moves_along_horizontal_wind_not_fixed_x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    scene = FakeUserScene()
    _, connectors = install_scene_geometry_recorders(simulation, monkeypatch)
    simulation.draw_wind_arrows(scene, [0, 2, 5], 0.5)
    np.testing.assert_allclose(connectors[0]["start"], [-10, -7, 0])


def test_vertical_only_visualization_has_no_horizontal_arrows() -> None:
    simulation = load_simulation_module()
    model = simulation.mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = simulation.mujoco.MjvScene(model, maxgeom=300)
    simulation.draw_wind_arrows(scene, [0, 0, 1], 0)
    assert scene.ngeom == 0


@pytest.mark.parametrize("name", ["figure8", "ellipse", "circle", "square"])
def test_predefined_trajectory_starts_at_supplied_xy_and_is_bounded(
    name: str,
) -> None:
    simulation = load_simulation_module()
    start_xy = np.array([2.0, -3.0])
    trajectory = simulation.PredefinedXYTrajectory.from_start(
        name=name,
        start_xy=start_xy,
        altitude=1.0,
        start_time=5.0,
        duration=20.0,
        size_m=1.0,
    )

    first = trajectory.sample(5.0)
    assert first is not None
    np.testing.assert_allclose(first.position_world, [2.0, -3.0, 1.0])
    assert trajectory.sample(25.0) is None

    samples = [
        trajectory.sample(5.0 + 19.999 * index / 100.0)
        for index in range(101)
    ]
    positions = np.array(
        [sample.position_world for sample in samples if sample is not None]
    )
    assert np.all(np.isfinite(positions))
    distance_from_start = np.linalg.norm(positions[:, :2] - start_xy, axis=1)
    assert float(np.max(distance_from_start)) <= 1.5


def test_square_is_continuous_at_segment_boundaries() -> None:
    simulation = load_simulation_module()
    trajectory = simulation.PredefinedXYTrajectory.from_start(
        name="square",
        start_xy=np.zeros(2),
        altitude=1.0,
        start_time=0.0,
        duration=20.0,
        size_m=1.0,
    )

    for boundary in (5.0, 10.0, 15.0):
        before = trajectory.sample(boundary - 1e-6)
        after = trajectory.sample(boundary + 1e-6)
        assert before is not None and after is not None
        assert np.linalg.norm(before.position_world - after.position_world) < 1e-5


def test_realtime_predefined_plot_shows_target_and_updates_actual() -> None:
    """Displays the complete target path and periodically redraws actual XY."""
    simulation = load_simulation_module()
    trajectory = simulation.PredefinedXYTrajectory.from_start(
        name="circle",
        start_xy=np.array([2.0, -3.0]),
        altitude=1.0,
        start_time=5.0,
        duration=20.0,
        size_m=1.0,
    )
    plot = simulation.create_realtime_predefined_trajectory_plot(trajectory)

    target_x, target_y = plot.target_line.get_data()
    assert len(target_x) == simulation.TRAJECTORY_PLOT_TARGET_SAMPLE_COUNT
    assert len(target_y) == simulation.TRAJECTORY_PLOT_TARGET_SAMPLE_COUNT
    np.testing.assert_allclose([target_x[0], target_y[0]], [2.0, -3.0])
    assert plot.target_line.get_label() == "target"
    assert plot.actual_line.get_label() == "actual"

    simulation.update_realtime_predefined_trajectory_plot(
        plot,
        position_world=np.array([2.0, -3.0, 1.0]),
        sim_time=5.0,
    )
    simulation.update_realtime_predefined_trajectory_plot(
        plot,
        position_world=np.array([2.1, -2.9, 1.0]),
        sim_time=5.1,
    )
    actual_x, actual_y = plot.actual_line.get_data()
    np.testing.assert_allclose(actual_x, [2.0])
    np.testing.assert_allclose(actual_y, [-3.0])

    simulation.update_realtime_predefined_trajectory_plot(
        plot,
        position_world=np.array([2.2, -2.8, 1.0]),
        sim_time=5.3,
    )
    actual_x, actual_y = plot.actual_line.get_data()
    np.testing.assert_allclose(actual_x, [2.0, 2.1, 2.2])
    np.testing.assert_allclose(actual_y, [-3.0, -2.9, -2.8])
    plot.close()


@pytest.mark.parametrize("name", ["unknown", "", "triangle"])
def test_predefined_trajectory_rejects_unknown_name(name: str) -> None:
    simulation = load_simulation_module()

    with pytest.raises(ValueError, match="unsupported predefined trajectory"):
        simulation.PredefinedXYTrajectory.from_start(
            name=name,
            start_xy=np.zeros(2),
            altitude=1.0,
            start_time=0.0,
            duration=20.0,
            size_m=1.0,
        )


def test_unlock_does_not_start_takeoff_or_offboard() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(
        initial_position_world=np.array([0.0, 0.0, 0.48]),
        takeoff_height_m=1.0,
    )

    actions = state.handle(
        simulation.UiCommand(simulation.UiCommandType.UNLOCK)
    )

    assert actions == [simulation.ControlAction.REQUEST_ARM]
    assert state.flight_state is simulation.FlightState.DISARMED
    assert not state.offboard
    assert state.target_position_world[2] == pytest.approx(0.48)


def test_takeoff_requires_confirmed_armed_state() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(np.array([1.0, -2.0, 0.48]), 1.0)

    assert state.handle(simulation.UiCommand(simulation.UiCommandType.TAKEOFF)) == []
    state.update_px4_status(armed=True, offboard=False)
    actions = state.handle(simulation.UiCommand(simulation.UiCommandType.TAKEOFF))

    assert actions == [
        simulation.ControlAction.ENTER_OFFBOARD,
        simulation.ControlAction.SEND_TAKEOFF_TARGET,
    ]
    assert state.flight_state is simulation.FlightState.TAKEOFF
    np.testing.assert_allclose(state.target_position_world, [1.0, -2.0, 1.48])


def test_trajectory_is_rejected_until_flying() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(np.array([0.0, 0.0, 0.48]), 1.0)

    actions = state.handle(
        simulation.UiCommand(
            simulation.UiCommandType.START_TRAJECTORY,
            "circle",
        )
    )

    assert actions == []
    assert state.active_trajectory_name == "none"


def test_move_position_updates_free_flight_target_by_exact_delta() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(
        np.array([0.0, 0.0, 0.48]),
        1.0,
    )
    state.flight_state = simulation.FlightState.FLYING

    actions = state.handle(
        simulation.UiCommand(
            simulation.UiCommandType.MOVE_POSITION,
            (0.5, 0.0, 0.0),
        )
    )

    assert actions == [simulation.ControlAction.SEND_POSITION_TARGET]
    np.testing.assert_allclose(
        state.target_position_world,
        [0.5, 0.0, 0.48],
    )


def test_move_position_accumulates_without_negative_z_clamp() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(
        np.array([0.0, 0.0, 0.48]),
        1.0,
    )
    state.flight_state = simulation.FlightState.FLYING
    command = simulation.UiCommand(
        simulation.UiCommandType.MOVE_POSITION,
        (0.0, 0.0, -0.5),
    )

    state.handle(command)
    state.handle(command)

    np.testing.assert_allclose(
        state.target_position_world,
        [0.0, 0.0, -0.52],
    )


@pytest.mark.parametrize(
    ("flight_state", "trajectory"),
    [
        ("ARMED", "none"),
        ("FLYING", "circle"),
    ],
)
def test_move_position_rejected_outside_free_flight(
    flight_state: str,
    trajectory: str,
) -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(
        np.array([1.0, 2.0, 3.0]),
        1.0,
    )
    state.flight_state = simulation.FlightState[flight_state]
    state.active_trajectory_name = trajectory
    initial_target = state.target_position_world.copy()

    actions = state.handle(
        simulation.UiCommand(
            simulation.UiCommandType.MOVE_POSITION,
            (0.5, 0.0, 0.0),
        )
    )

    assert actions == []
    np.testing.assert_allclose(state.target_position_world, initial_target)


def test_landing_cancels_active_trajectory() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(np.array([0.0, 0.0, 0.48]), 1.0)
    state.flight_state = simulation.FlightState.FLYING
    state.active_trajectory_name = "circle"

    actions = state.handle(simulation.UiCommand(simulation.UiCommandType.LAND))

    assert state.flight_state is simulation.FlightState.LANDING
    assert state.active_trajectory_name == "none"
    assert actions == [simulation.ControlAction.SEND_LANDING_TARGET]


def test_emergency_stop_latches_and_requests_zero_thrust_and_disarm() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(np.array([0.0, 0.0, 0.48]), 1.0)

    actions = state.handle(
        simulation.UiCommand(simulation.UiCommandType.EMERGENCY_STOP)
    )

    assert state.flight_state is simulation.FlightState.EMERGENCY_STOP
    assert actions == [
        simulation.ControlAction.ZERO_THRUST,
        simulation.ControlAction.REQUEST_DISARM,
    ]
    assert state.handle(simulation.UiCommand(simulation.UiCommandType.TAKEOFF)) == []


def test_wind_toggle_is_independent_of_normal_flight_state() -> None:
    simulation = load_simulation_module()
    state = simulation.FlightControlState(np.array([0.0, 0.0, 0.48]), 1.0)
    state.wind_configured = True

    state.handle(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))

    assert state.wind_enabled
    assert state.flight_state is simulation.FlightState.DISARMED


def test_motor_outputs_are_clipped_and_scaled() -> None:
    simulation = load_simulation_module()

    thrusts = simulation.motor_outputs_to_thrusts(
        normalized_outputs=[-0.2, 0.25, 0.5, 1.2],
        rotor_thrust_scales=np.full(4, 20.0),
    )

    np.testing.assert_allclose(thrusts, [0.0, 5.0, 10.0, 20.0])


def test_hover_matched_thrust_balances_vehicle_weight() -> None:
    simulation = load_simulation_module()

    total_max_thrust = simulation.compute_hover_matched_total_max_thrust(
        mass_kg=8.0,
        hover_thrust=0.4,
    )

    assert total_max_thrust * 0.4 == pytest.approx(8.0 * 9.80665)


def test_requested_xml_resolves_required_model_bindings() -> None:
    simulation = load_simulation_module()
    model = simulation.mujoco.MjModel.from_xml_path(
        str(simulation.default_xml_path())
    )

    bindings = simulation.ModelBindings.resolve(model)

    assert bindings.quadrotor_body_id > 0
    assert bindings.main_joint_id >= 0
    assert bindings.main_qpos_adr >= 0
    assert bindings.main_dof_adr >= 0
    assert bindings.rotor_thrust_actuator_ids.shape == (4,)
    assert bindings.rotor_spin_actuator_ids.shape == (4,)
    assert bindings.uav_mass_kg > 0.0


def test_default_parameter_path_contains_hover_thrust() -> None:
    simulation = load_simulation_module()

    hover_thrust = simulation.read_px4_param(
        simulation.default_params_path(),
        "MPC_THR_HOVER",
        default=0.5,
    )

    assert 0.0 < hover_thrust < 1.0


class FakeMavSender:
    """Records MAVLink send calls made by ``Px4Bridge``."""

    def __init__(self) -> None:
        self.command_long_calls: list[tuple] = []
        self.heartbeat_calls: list[tuple] = []
        self.hil_sensor_calls: list[tuple] = []
        self.hil_state_calls: list[tuple] = []
        self.position_target_calls: list[tuple] = []

    def command_long_send(self, *args) -> None:
        self.command_long_calls.append(args)

    def heartbeat_send(self, *args) -> None:
        self.heartbeat_calls.append(args)

    def hil_sensor_send(self, *args) -> None:
        self.hil_sensor_calls.append(args)

    def hil_state_quaternion_send(self, *args) -> None:
        self.hil_state_calls.append(args)

    def set_position_target_local_ned_send(self, *args) -> None:
        self.position_target_calls.append(args)


class FakeMavlinkConnection:
    """Non-blocking receive queue plus a recording MAVLink sender."""

    def __init__(self) -> None:
        self.mav = FakeMavSender()
        self.messages: list[object] = []
        self.closed = False
        self.command_long_calls: list[tuple[Any, ...]] = []

    def write(self, packet: bytes) -> int:
        """Decode actual outgoing MAVLink packets in an in-memory transport.

        Args:
            packet: Bytes produced by the production MAVLink 2 encoder.

        Returns:
            Number of bytes consumed by the transport.
        """
        from pymavlink.dialects.v20 import ardupilotmega

        messages = ardupilotmega.MAVLink(None).parse_buffer(packet) or []
        for message in messages:
            if message.get_type() == "COMMAND_LONG":
                self.command_long_calls.append(tuple(
                    getattr(message, field) for field in message.get_fieldnames()
                ))
        return len(packet)

    def recv_match(self, **_kwargs):
        if not self.messages:
            return None
        return self.messages.pop(0)

    def close(self) -> None:
        self.closed = True


def make_fake_bridge(simulation, clock):
    """Creates a connected bridge without opening sockets."""
    connections: list[FakeMavlinkConnection] = []

    def connection_factory(*_args, **_kwargs):
        connection = FakeMavlinkConnection()
        connections.append(connection)
        return connection

    bridge = simulation.Px4Bridge(
        rotor_thrust_scales=np.full(4, 20.0),
        connection_factory=connection_factory,
        clock=clock,
    )
    bridge.connect()
    return bridge, connections


def test_request_arm_sends_only_arm_command() -> None:
    simulation = load_simulation_module()
    bridge, connections = make_fake_bridge(simulation, lambda: 10.0)

    bridge.request_arm()
    bridge.request_arm()

    calls = connections[1].command_long_calls
    assert len(calls) == 1
    assert calls[0][2] == simulation.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert calls[0][4] == pytest.approx(1.0)


def test_request_offboard_sends_only_mode_command() -> None:
    simulation = load_simulation_module()
    bridge, connections = make_fake_bridge(simulation, lambda: 10.0)

    bridge.request_offboard()

    calls = connections[1].command_long_calls
    assert len(calls) == 1
    assert calls[0][2] == simulation.mavutil.mavlink.MAV_CMD_DO_SET_MODE
    assert calls[0][4] == pytest.approx(29.0)
    assert calls[0][5] == pytest.approx(6.0)


def test_request_shutdown_sends_one_px4_autopilot_shutdown_command() -> None:
    simulation = load_simulation_module()
    bridge, connections = make_fake_bridge(simulation, lambda: 10.0)

    bridge.request_shutdown()
    bridge.request_shutdown()

    calls = connections[1].command_long_calls
    assert len(calls) == 1
    assert (
        calls[0][2]
        == simulation.mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
    )
    assert calls[0][4] == pytest.approx(2.0)
    assert calls[0][5:11] == pytest.approx((0.0,) * 6)


def test_world_to_px4_conversion_flips_y_and_z() -> None:
    simulation = load_simulation_module()

    converted = simulation.Px4Bridge.mujoco_world_to_px4_local(
        np.array([1.0, 2.0, 3.0])
    )

    np.testing.assert_allclose(converted, [1.0, -2.0, -3.0])


def test_control_becomes_stale_after_timeout() -> None:
    simulation = load_simulation_module()
    clock_state = {"value": 10.0}
    bridge, _connections = make_fake_bridge(
        simulation,
        lambda: clock_state["value"],
    )
    bridge.last_control_received_wall_time = 10.0
    clock_state["value"] = 10.51

    assert bridge.control_is_stale(max_age_s=0.5)


def test_receive_control_maps_latest_actuator_message() -> None:
    simulation = load_simulation_module()
    bridge, connections = make_fake_bridge(simulation, lambda: 10.0)
    connections[0].messages.append(
        SimpleNamespace(
            controls=[0.0, 0.25, 0.5, 1.0] + [0.0] * 12,
            get_type=lambda: "HIL_ACTUATOR_CONTROLS",
        )
    )

    thrusts = bridge.receive_rotor_thrusts()

    np.testing.assert_allclose(thrusts, [0.0, 5.0, 10.0, 20.0])
    assert bridge.last_control_received_wall_time == pytest.approx(10.0)


def test_control_panel_callbacks_enqueue_typed_commands() -> None:
    simulation = load_simulation_module()
    command_queue: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(command_queue)
    panel.update(make_panel_snapshot(simulation))

    panel._on_unlock(None)
    panel._on_toggle_wind(None)

    unlock_command = command_queue.get_nowait()
    wind_command = command_queue.get_nowait()
    assert unlock_command == simulation.UiCommand(simulation.UiCommandType.UNLOCK)
    assert wind_command == simulation.UiCommand(
        simulation.UiCommandType.TOGGLE_WIND
    )
    panel.close()


def test_control_panel_uses_compact_horizontal_proportions() -> None:
    """Keeps the control window narrower without reducing its height."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())

    np.testing.assert_allclose(panel.figure.get_size_inches(), [7.4, 5.0])
    assert panel.buttons["emergency"].label.get_text() == "Emergency"
    panel.close()


def test_xyz_buttons_enqueue_exact_half_metre_world_deltas() -> None:
    simulation = load_simulation_module()
    command_queue: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(command_queue)
    expected_deltas = {
        "x_minus": (-0.5, 0.0, 0.0),
        "x_plus": (0.5, 0.0, 0.0),
        "y_minus": (0.0, -0.5, 0.0),
        "y_plus": (0.0, 0.5, 0.0),
        "z_minus": (0.0, 0.0, -0.5),
        "z_plus": (0.0, 0.0, 0.5),
    }

    for control_name in expected_deltas:
        control = panel.triangle_controls[control_name]
        display_x, display_y = control.axis.transAxes.transform((0.5, 0.5))
        event = MouseEvent(
            "button_press_event",
            panel.figure.canvas,
            display_x,
            display_y,
            button=1,
        )
        panel.figure.canvas.callbacks.process("button_press_event", event)

    commands = [
        command_queue.get_nowait()
        for _control_name in expected_deltas
    ]
    assert [command.command_type for command in commands] == [
        simulation.UiCommandType.MOVE_POSITION
    ] * 6
    assert [command.payload for command in commands] == list(
        expected_deltas.values()
    )
    panel.close()


def test_xyz_triangle_controls_match_reference_cross_layout() -> None:
    """Places XY as a cross and the Z pair beneath it."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())
    controls = panel.triangle_controls

    def center(name: str) -> tuple[float, float]:
        bounds = controls[name].axis.get_position().bounds
        return bounds[0] + 0.5 * bounds[2], bounds[1] + 0.5 * bounds[3]

    x_plus = center("x_plus")
    x_minus = center("x_minus")
    y_plus = center("y_plus")
    y_minus = center("y_minus")
    z_plus = center("z_plus")
    z_minus = center("z_minus")

    assert x_plus[0] == pytest.approx(x_minus[0])
    assert y_plus[1] == pytest.approx(y_minus[1])
    assert x_plus[1] > y_plus[1] > x_minus[1]
    assert y_plus[0] < x_plus[0] < y_minus[0]
    assert z_plus[1] == pytest.approx(z_minus[1])
    assert z_plus[1] < x_minus[1]
    assert controls["x_plus"].pointer == "up"
    assert controls["x_minus"].pointer == "down"
    assert controls["y_plus"].pointer == "left"
    assert controls["y_minus"].pointer == "right"
    assert controls["z_plus"].pointer == "up"
    assert controls["z_minus"].pointer == "down"
    panel.close()


def test_control_panel_trajectory_callback_carries_name() -> None:
    simulation = load_simulation_module()
    command_queue: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(command_queue)

    panel._on_trajectory("figure8")

    assert command_queue.get_nowait() == simulation.UiCommand(
        simulation.UiCommandType.START_TRAJECTORY,
        "figure8",
    )
    panel.close()


def test_control_panel_shows_enabled_for_zero_configured_wind() -> None:
    simulation = load_simulation_module()
    command_queue: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(command_queue)
    snapshot = simulation.PanelSnapshot(
        px4_connected=True,
        armed=False,
        offboard=False,
        flight_state="DISARMED",
        trajectory="none",
        position_world=np.zeros(3),
        target_world=np.zeros(3),
        configured_wind_world=np.zeros(3),
        active_wind_world=np.zeros(3),
        filtered_residual_acceleration=np.zeros(3),
        pinn_force_world=np.zeros(3),
        wind_enabled=True,
    )

    panel.update(snapshot)

    assert panel.buttons["wind"].label.get_text() == "Wind ON"
    panel.close()


STATE_DRIVEN_BUTTON_NAMES = (
    "unlock",
    "takeoff",
    "land",
    "wind",
    "figure8",
    "ellipse",
    "circle",
    "square",
)


def make_panel_snapshot(
    simulation: Any,
    **overrides: object,
) -> Any:
    """Builds a valid panel snapshot with focused state overrides."""
    values: dict[str, object] = {
        "px4_connected": True,
        "armed": False,
        "offboard": False,
        "flight_state": "DISARMED",
        "trajectory": "none",
        "position_world": np.zeros(3),
        "target_world": np.zeros(3),
        "configured_wind_world": np.array([8.0, 0.0, 0.0]),
        "active_wind_world": np.zeros(3),
        "filtered_residual_acceleration": np.array([0.2, -0.1, 0.05]),
        "pinn_force_world": np.array([0.5, -0.25, 0.125]),
        "wind_enabled": False,
    }
    values.update(overrides)
    return simulation.PanelSnapshot(**values)


@pytest.mark.parametrize(
    ("snapshot_overrides", "active_button_name"),
    [
        ({"armed": True}, "unlock"),
        ({"flight_state": "TAKEOFF"}, "takeoff"),
        ({"flight_state": "LANDING"}, "land"),
        ({"wind_enabled": True}, "wind"),
        ({"trajectory": "figure8"}, "figure8"),
        ({"trajectory": "ellipse"}, "ellipse"),
        ({"trajectory": "circle"}, "circle"),
        ({"trajectory": "square"}, "square"),
    ],
)
def test_panel_state_highlight_matches_confirmed_state(
    snapshot_overrides: dict[str, object],
    active_button_name: str,
) -> None:
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())

    panel.update(make_panel_snapshot(simulation, **snapshot_overrides))

    for button_name in STATE_DRIVEN_BUTTON_NAMES:
        button = panel.buttons[button_name]
        expected_color = (
            "#b7d7f0"
            if button_name == active_button_name
            else "#e2e2e2"
        )
        assert button.color == expected_color
        np.testing.assert_allclose(
            button.ax.get_facecolor(),
            matplotlib_colors.to_rgba(expected_color),
        )
    assert panel.buttons["emergency"].color == "#ffb3b3"
    np.testing.assert_allclose(
        panel.buttons["emergency"].ax.get_facecolor(),
        matplotlib_colors.to_rgba("#ffb3b3"),
    )
    panel.close()


def test_panel_state_highlight_resets_when_state_becomes_inactive() -> None:
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())
    panel.update(make_panel_snapshot(simulation, trajectory="circle"))

    panel.update(make_panel_snapshot(simulation))

    for button_name in STATE_DRIVEN_BUTTON_NAMES:
        assert panel.buttons[button_name].color == "#e2e2e2"
    panel.close()


def test_panel_unlock_pending_uses_distinct_feedback_color() -> None:
    """Distinguishes a sent arm request from confirmed PX4 armed state."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())

    panel.update(make_panel_snapshot(simulation, unlock_pending=True))

    unlock_button = panel.buttons["unlock"]
    assert unlock_button.color == "#fde68a"
    np.testing.assert_allclose(
        unlock_button.ax.get_facecolor(),
        matplotlib_colors.to_rgba("#fde68a"),
    )
    panel.close()


def test_control_panel_keeps_status_and_state_button_updates() -> None:
    """Preserve the diagnostics panel added after the original test suite."""
    simulation = load_simulation_module()
    panel = simulation.ControlPanel(queue.SimpleQueue())

    panel.update(make_panel_snapshot(simulation, wind_enabled=True))

    assert "PX4:" in panel.status_text.get_text()
    assert panel.buttons["wind"].label.get_text() == "Wind ON"
    panel.close()


def test_argument_parser_defaults_match_approved_design() -> None:
    simulation = load_simulation_module()

    config = simulation.parse_config([])

    assert config.wind_velocity_world is None
    assert not hasattr(config, "drag_coefficients_world")
    assert config.takeoff_height_m == pytest.approx(1.0)
    assert config.sim_tcp_port == 4560
    assert config.command_udp_port == 14580
    assert config.viewer_enabled
    assert not hasattr(config, "csv_output_dir")
    with pytest.raises(SystemExit):
        simulation.parse_config(["--drag-x", "0.1"])
    with pytest.raises(SystemExit):
        simulation.parse_config(["--csv-output-dir", "/tmp/wind"])


def test_apply_external_wind_force_clears_stale_torque() -> None:
    """Applies only the requested force and clears prior torque."""
    simulation = load_simulation_module()
    data = SimpleNamespace(xfrc_applied=np.ones((3, 6), dtype=float))

    simulation.apply_external_wind_force(
        data=data,
        body_id=1,
        force_world=np.array([0.32, -0.02, 0.01]),
    )

    np.testing.assert_allclose(
        data.xfrc_applied[1],
        [0.32, -0.02, 0.01, 0.0, 0.0, 0.0],
    )


def test_stale_flight_control_zeroes_rotors_and_latches_emergency() -> None:
    simulation = load_simulation_module()
    data = SimpleNamespace(ctrl=np.ones(12, dtype=float))
    bindings = SimpleNamespace(
        rotor_thrust_actuator_ids=np.array([0, 1, 2, 3], dtype=int)
    )
    flight_control = simulation.FlightControlState(
        np.array([0.0, 0.0, 0.48]),
        1.0,
    )
    flight_control.flight_state = simulation.FlightState.FLYING
    bridge = SimpleNamespace(
        control_is_stale=lambda max_age_s: max_age_s == pytest.approx(0.5),
        request_disarm_calls=0,
    )

    def request_disarm() -> None:
        bridge.request_disarm_calls += 1

    bridge.request_disarm = request_disarm

    triggered = simulation.enforce_stale_control_safety(
        data=data,
        bindings=bindings,
        flight_control=flight_control,
        bridge=bridge,
        max_age_s=0.5,
    )

    assert triggered
    np.testing.assert_allclose(data.ctrl[:4], np.zeros(4))
    assert flight_control.flight_state is simulation.FlightState.EMERGENCY_STOP
    assert bridge.request_disarm_calls == 1


class RecordingBridge:
    """Minimal bridge double for application command dispatch."""

    def __init__(self) -> None:
        self.armed = False
        self.offboard = False
        self.connected = True
        self.arm_calls = 0
        self.offboard_calls = 0
        self.disarm_calls = 0
        self.shutdown_calls = 0
        self.targets: list[np.ndarray] = []
        self.state_calls = 0
        self.last_control_received_wall_time = 0.0
        self.arm_ack_sequence = 0
        self.last_arm_ack_result: int | None = None

    def request_arm(self) -> None:
        self.arm_calls += 1

    def request_offboard(self, force: bool = False) -> None:
        del force
        self.offboard_calls += 1

    def request_disarm(self) -> None:
        self.disarm_calls += 1

    def request_shutdown(self) -> None:
        self.shutdown_calls += 1

    def send_position_target(self, target_world, target_yaw_rad: float) -> None:
        del target_yaw_rad
        self.targets.append(np.asarray(target_world, dtype=float).copy())

    def send_state(self, **_kwargs) -> None:
        self.state_calls += 1

    def send_command_heartbeat(self) -> None:
        pass

    def poll_status(self) -> None:
        pass

    def receive_rotor_thrusts(self) -> np.ndarray:
        return np.zeros(4)

    def control_is_stale(self, _max_age_s: float) -> bool:
        return False

    def heartbeat_is_recent(self) -> bool:
        return True


class RecordingPinnPredictor:
    """Records state inputs and returns a configured residual acceleration."""

    def __init__(self, prediction: np.ndarray | None = None) -> None:
        self.prediction = np.asarray(
            prediction if prediction is not None else [4.0, -2.0, 1.0],
            dtype=float,
        )
        self.calls: list[dict[str, np.ndarray]] = []

    def predict(self, **state: np.ndarray) -> np.ndarray:
        """Records one complete PINN state and returns the configured value."""
        self.calls.append(
            {
                key: np.asarray(value, dtype=float).copy()
                for key, value in state.items()
            }
        )
        return self.prediction.copy()


@pytest.mark.parametrize("raw_vertical", [-1.0, 1.0])
def test_horizontal_wind_clears_vertical_force_and_stale_filter(
    raw_vertical: float,
) -> None:
    """Project applied wind to XY while preserving raw predictions and XY EMA."""
    simulation = load_simulation_module()
    predictor = RecordingPinnPredictor(np.array([4.0, -2.0, raw_vertical]))
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=predictor,
    )
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
    app.residual_filter.filtered[:] = [2.0, -1.0, 1.0]
    active, _relative, force = app._apply_wind_force()

    np.testing.assert_allclose(active, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        app.last_raw_residual_acceleration, [4.0, -2.0, raw_vertical],
    )
    np.testing.assert_allclose(app.residual_filter.filtered, [2.1, -1.05, 0.0])
    np.testing.assert_allclose(
        app.last_filtered_residual_acceleration, [2.1, -1.05, 0.0],
    )
    assert force[2] == 0.0
    assert app.last_pinn_force[2] == 0.0
    assert app.data.xfrc_applied[app.bindings.quadrotor_body_id, 2] == 0.0
    assert app._panel_snapshot(app._current_state()[0]).pinn_force_world[2] == 0.0


@pytest.mark.parametrize("vertical_velocity", [0.0, 0.5])
def test_real_pinn_horizontal_wind_applies_no_vertical_force(
    vertical_velocity: float,
) -> None:
    """Real checkpoint outputs must not inject Fz, even during vertical motion."""
    simulation = load_simulation_module()
    simulation.torch.manual_seed(0)
    predictor = simulation.PinnWindResidualModel(
        device=simulation.torch.device("cpu"),
    )
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(), pinn_predictor=predictor,
    )
    app.data.qvel[app.bindings.main_dof_adr + 2] = vertical_velocity
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TOGGLE_WIND))
    _active, _relative, force = app._apply_wind_force()
    assert np.isfinite(app.last_raw_residual_acceleration).all()
    assert force[2] == 0.0
    assert app.data.xfrc_applied[app.bindings.quadrotor_body_id, 2] == 0.0


def test_application_pinn_force_respects_panel_toggle() -> None:
    """Uses PINN only while enabled and converts all axes through mass."""
    simulation = load_simulation_module()
    predictor = RecordingPinnPredictor(np.array([4.0, -2.0, 1.0]))
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(),
        pinn_predictor=predictor,
    )

    active_wind, relative_velocity, force = app._apply_wind_force()
    np.testing.assert_allclose(active_wind, np.zeros(3))
    np.testing.assert_allclose(relative_velocity, app._current_state()[1])
    np.testing.assert_allclose(force, np.zeros(3))
    assert predictor.calls == []

    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app.flight_control.wind_enabled = True
    active_wind, relative_velocity, force = app._apply_wind_force()

    np.testing.assert_allclose(active_wind, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(relative_velocity, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        predictor.calls[0]["quaternion_wxyz"],
        app._current_state()[2],
    )
    np.testing.assert_allclose(
        app.last_filtered_residual_acceleration,
        [0.2, -0.1, 0.0],
    )
    np.testing.assert_allclose(
        force,
        app.bindings.uav_mass_kg
        * app.last_filtered_residual_acceleration,
    )
    np.testing.assert_allclose(
        app.data.xfrc_applied[app.bindings.quadrotor_body_id, 0:3],
        force,
    )
    assert force[2] == 0.0


def test_wind_disable_resets_pinn_force_and_filter() -> None:
    """Clears stale PINN state and wrench when wind is switched off."""
    simulation = load_simulation_module()
    predictor = RecordingPinnPredictor(np.array([4.0, -2.0, 1.0]))
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(),
        pinn_predictor=predictor,
    )
    app._process_ui_command(simulation.UiCommand(
        simulation.UiCommandType.SET_WIND, (1.0, 0.0, 0.0),
    ))
    app.flight_control.wind_enabled = True
    app._apply_wind_force()

    app.flight_control.wind_enabled = False
    active_wind, _relative_velocity, force = app._apply_wind_force()

    np.testing.assert_allclose(active_wind, np.zeros(3))
    np.testing.assert_allclose(force, np.zeros(3))
    np.testing.assert_allclose(app.residual_filter.filtered, np.zeros(3))
    np.testing.assert_allclose(
        app.last_raw_residual_acceleration,
        np.zeros(3),
    )
    np.testing.assert_allclose(
        app.last_filtered_residual_acceleration,
        np.zeros(3),
    )
    np.testing.assert_allclose(app.last_pinn_force, np.zeros(3))
    np.testing.assert_allclose(
        app.data.xfrc_applied[app.bindings.quadrotor_body_id],
        np.zeros(6),
    )


def test_application_dispatches_unlock_and_takeoff_separately() -> None:
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )

    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.UNLOCK))
    assert bridge.arm_calls == 1
    assert bridge.offboard_calls == 0

    app.flight_control.update_px4_status(armed=True, offboard=False)
    app._process_ui_command(simulation.UiCommand(simulation.UiCommandType.TAKEOFF))
    assert bridge.offboard_calls == 1
    assert len(bridge.targets) == 1


def test_rejected_arm_ack_clears_pending_request_and_reports_reason() -> None:
    """Stops blind retries after PX4 explicitly rejects the arm command."""
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    bridge.last_command_ack = (
        f"{simulation.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM}:1"
    )
    bridge.last_statustext = "Arming denied: resolve system health failures first"
    bridge.arm_ack_sequence = 1
    bridge.last_arm_ack_result = 1
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.unlock_requested = True

    app._update_unlock_request_state()

    assert not app.unlock_requested
    assert "PX4 arm rejected" in app.last_error
    assert bridge.last_statustext in app.last_error


def test_application_sends_accepted_move_target_through_px4_bridge() -> None:
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.flight_control.flight_state = simulation.FlightState.FLYING
    initial_target = app.flight_control.target_position_world.copy()

    app._process_ui_command(
        simulation.UiCommand(
            simulation.UiCommandType.MOVE_POSITION,
            (0.0, 0.5, 0.0),
        )
    )

    assert len(bridge.targets) == 1
    np.testing.assert_allclose(
        bridge.targets[0],
        initial_target + np.array([0.0, 0.5, 0.0]),
    )


def test_application_trajectory_command_opens_realtime_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connects an accepted predefined-trajectory command to its live plot."""
    simulation = load_simulation_module()
    fake_plot = SimpleNamespace(figure=object())
    created_trajectories: list[Any] = []
    shown_figures: list[object] = []

    def create_plot(trajectory: Any) -> Any:
        created_trajectories.append(trajectory)
        return fake_plot

    monkeypatch.setattr(
        simulation,
        "create_realtime_predefined_trajectory_plot",
        create_plot,
    )
    monkeypatch.setattr(
        simulation,
        "show_realtime_figure",
        shown_figures.append,
    )
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=RecordingBridge(),
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.flight_control.flight_state = simulation.FlightState.FLYING

    app._process_ui_command(
        simulation.UiCommand(
            simulation.UiCommandType.START_TRAJECTORY,
            "figure8",
        )
    )

    assert len(created_trajectories) == 1
    assert created_trajectories[0] is app.active_trajectory
    assert app.trajectory_plot is fake_plot
    assert shown_figures == [fake_plot.figure]


def test_exit_button_enqueues_typed_exit_command() -> None:
    simulation = load_simulation_module()
    command_queue: queue.SimpleQueue = queue.SimpleQueue()
    panel = simulation.ControlPanel(command_queue)

    panel.buttons["exit"]._observers.process("clicked", None)

    assert command_queue.get_nowait() == simulation.UiCommand(
        simulation.UiCommandType.EXIT
    )
    panel.close()


def test_application_exit_is_safe_and_idempotent() -> None:
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.data.ctrl[app.bindings.rotor_thrust_actuator_ids] = 1.0
    command = simulation.UiCommand(simulation.UiCommandType.EXIT)

    app._process_ui_command(command)
    app._process_ui_command(command)

    assert app.exit_requested
    np.testing.assert_allclose(
        app.data.ctrl[app.bindings.rotor_thrust_actuator_ids],
        np.zeros(4),
    )
    assert bridge.disarm_calls == 1
    assert bridge.shutdown_calls == 1


def test_exit_stops_iteration_before_rotor_output_is_reapplied() -> None:
    simulation = load_simulation_module()

    class NoRotorAfterExitBridge(RecordingBridge):
        """Fails if the iteration requests rotor output after Exit."""

        def receive_rotor_thrusts(self) -> np.ndarray:
            raise AssertionError("rotor output requested after Exit")

    bridge = NoRotorAfterExitBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.command_queue.put(
        simulation.UiCommand(simulation.UiCommandType.EXIT)
    )

    iteration_completed = app._run_iteration()

    assert iteration_completed is False
    assert app.exit_requested
    assert bridge.shutdown_calls == 1


def test_application_iteration_advances_mujoco_without_px4_sockets() -> None:
    simulation = load_simulation_module()
    bridge = RecordingBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config(["--no-viewer"]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    initial_time = float(app.data.time)

    app._run_iteration()

    assert app.data.time == pytest.approx(initial_time + app.model.opt.timestep)
    assert bridge.state_calls == 1
    np.testing.assert_allclose(
        app.data.ctrl[app.bindings.rotor_thrust_actuator_ids],
        np.zeros(4),
    )


def test_run_draws_active_wind_before_viewer_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    events: list[object] = []
    user_scene = object()

    class NullContext:
        """No-op context manager used by the fake viewer lock."""

        def __enter__(self) -> None:
            return None

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            del exc_type, exc_value, traceback
            return False

    class FakeViewer:
        """Records viewer synchronization without opening a window."""

        def __init__(self) -> None:
            self.user_scn = user_scene

        def lock(self) -> NullContext:
            return NullContext()

        def sync(self) -> None:
            events.append("sync")

        def close(self) -> None:
            pass

    class FakePanel:
        """Supplies the panel interface used by the application run loop."""

        def __init__(
            self, _command_queue: queue.SimpleQueue,
            initial_wind_world: Any = None,
        ) -> None:
            self.figure = SimpleNamespace(show=lambda: None)

        def close(self) -> None:
            pass

    bridge = RecordingBridge()
    config = simulation.parse_config([])
    app = simulation.WindSimulationApp(
        config,
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.last_active_wind = np.array([8.0, 0.0, 0.0])
    app.data.time = 2.0
    app.clock = lambda: 0.0
    app.sleep = lambda _duration: None
    runtime_checks = iter([True, False])
    monkeypatch.setattr(
        app,
        "_runtime_is_open",
        lambda: next(runtime_checks),
    )
    monkeypatch.setattr(app, "_run_iteration", lambda: True)
    monkeypatch.setattr(simulation, "ControlPanel", FakePanel)
    monkeypatch.setattr(simulation.plt, "ion", lambda: None)
    fake_viewer = FakeViewer()
    monkeypatch.setattr(
        simulation,
        "ManagedPassiveViewer",
        lambda _model, _data: fake_viewer,
    )
    monkeypatch.setattr(
        simulation,
        "draw_wind_arrows",
        lambda scene, wind, simulation_time_s: events.append(
            (scene, np.asarray(wind).copy(), simulation_time_s)
        ),
    )

    app.run()

    draw_event = events[0]
    assert draw_event[0] is user_scene
    np.testing.assert_allclose(draw_event[1], [8.0, 0.0, 0.0])
    assert draw_event[2] == pytest.approx(2.0)
    assert events[1] == "sync"


def test_run_exit_closes_resources_without_viewer_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = load_simulation_module()
    events: list[str] = []

    class NullContext:
        """No-op context manager for the fake viewer lock."""

        def __enter__(self) -> None:
            return None

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            del exc_type, exc_value, traceback
            return False

    class FakeViewer:
        """Records synchronization and closure."""

        user_scn = object()

        def lock(self) -> NullContext:
            return NullContext()

        def sync(self) -> None:
            events.append("viewer_sync")

        def close(self) -> None:
            events.append("viewer_close")

    class FakePanel:
        """Records panel closure without creating a second figure."""

        def __init__(
            self, _command_queue: queue.SimpleQueue,
            initial_wind_world: Any = None,
        ) -> None:
            self.figure = SimpleNamespace(show=lambda: None)

        def close(self) -> None:
            events.append("panel_close")

    bridge = RecordingBridge()
    app = simulation.WindSimulationApp(
        simulation.parse_config([]),
        bridge=bridge,
        pinn_predictor=RecordingPinnPredictor(),
    )
    app.command_queue.put(
        simulation.UiCommand(simulation.UiCommandType.EXIT)
    )
    app.clock = lambda: 0.0
    app.sleep = lambda _duration: None
    runtime_checks = iter([True, False])
    monkeypatch.setattr(
        app,
        "_runtime_is_open",
        lambda: next(runtime_checks),
    )
    monkeypatch.setattr(simulation, "ControlPanel", FakePanel)
    monkeypatch.setattr(simulation.plt, "ion", lambda: None)
    fake_viewer = FakeViewer()
    monkeypatch.setattr(
        simulation,
        "ManagedPassiveViewer",
        lambda _model, _data: fake_viewer,
    )
    monkeypatch.setattr(
        simulation,
        "draw_wind_arrows",
        lambda *_args: events.append("draw"),
    )

    app.run()

    assert app.exit_requested
    assert bridge.disarm_calls == 1
    assert bridge.shutdown_calls == 1
    assert "draw" not in events
    assert "viewer_sync" not in events
    assert events[-2:] == ["viewer_close", "panel_close"]
