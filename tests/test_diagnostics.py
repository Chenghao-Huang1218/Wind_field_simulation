"""Offline regressions for PX4 wind-panel diagnostics, without sockets."""

from __future__ import annotations

import io
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from wind_field_simulation import simulation as wind
from pymavlink.dialects.v10 import ardupilotmega as mavlink1  # noqa: E402
from pymavlink.dialects.v20 import ardupilotmega as mavlink2  # noqa: E402


def test_panel_heartbeat_enables_px4_status_text() -> None:
    """PX4's STATUSTEXT stream requires a GCS heartbeat on this link."""
    wire = io.BytesIO()
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: 10.0)
    bridge.command_link = SimpleNamespace(mav=wind.mavutil.mavlink.MAVLink(wire))

    bridge.send_command_heartbeat()

    parser = wind.mavutil.mavlink.MAVLink(None)
    message = parser.parse_char(wire.getvalue())
    assert message.get_type() == "HEARTBEAT"
    assert message.type == 6  # MAV_TYPE_GCS, as consumed by PX4's receiver.


def test_status_text_is_visible_when_it_arrives_after_rejection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Delayed PX4 text must reach the terminal even after the ACK is handled."""
    messages: deque[Any] = deque()

    def receive(**_kwargs: Any) -> Any | None:
        """Returns queued messages without opening a transport."""
        return messages.popleft() if messages else None

    bridge = wind.Px4Bridge([1.0] * 4)
    bridge.command_link = SimpleNamespace(recv_match=receive)
    app = wind.WindSimulationApp.__new__(wind.WindSimulationApp)
    app.bridge = bridge
    app.flight_control = SimpleNamespace(armed=False)
    app.unlock_requested = True
    app.last_processed_command_ack = "none"
    app.last_processed_arm_ack_sequence = 0
    app.unlock_request_started_wall_time = 10.0
    app.unlock_ack_accepted = False
    app.clock = lambda: 10.0

    messages.append(wind.mavutil.mavlink.MAVLink_command_ack_message(400, 1))
    bridge.poll_status()
    app._update_unlock_request_state()
    rejection = capsys.readouterr().out
    assert "MAV_RESULT_TEMPORARILY_REJECTED" in rejection
    assert ": none" not in rejection
    assert not app.unlock_requested

    messages.append(
        wind.mavutil.mavlink.MAVLink_statustext_message(
            2, b"Arming denied: Resolve system health failures first"
        )
    )
    bridge.poll_status()
    assert "Arming denied: Resolve system health failures first" in (
        capsys.readouterr().out
    )
    assert bridge.last_statustext.startswith("Arming denied:")


def test_command_link_preserves_mavlink2_chunk_metadata() -> None:
    """The first outgoing heartbeat must not negotiate away extension fields."""
    wire = io.BytesIO()

    def connection_factory(*_args: Any, **_kwargs: Any) -> Any:
        """Substitutes an in-memory wire for the TCP/UDP connections."""
        return SimpleNamespace(mav=mavlink1.MAVLink(wire), write=wire.write)

    bridge = wind.Px4Bridge(
        [1.0] * 4, connection_factory=connection_factory, clock=lambda: 10.0
    )
    bridge.connect()
    bridge.send_command_heartbeat()
    assert wire.getvalue()[0] == 253  # MAVLink 2 framing magic.


@pytest.mark.parametrize("message_id", [0, 17])
def test_statustext_chunks_form_one_complete_reason(
    message_id: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """PX4 starts its text ID counter at zero; both IDs must reassemble."""
    reason = b"Arming denied: Resolve system health failures first"
    assert len(reason) == 51
    messages = deque(
        [mavlink2.MAVLink_statustext_message(2, reason[:50], message_id, 0)]
    )
    bridge = wind.Px4Bridge([1.0] * 4)
    bridge.command_link = SimpleNamespace(
        recv_match=lambda **_kwargs: messages.popleft() if messages else None
    )
    bridge.poll_status()
    # Preserve the prefix immediately: PX4 also emits exactly-50-byte texts
    # without a terminating extra chunk. A later suffix must extend it.
    assert bridge.last_statustext == reason[:50].decode("ascii")
    capsys.readouterr()

    messages.append(
        mavlink2.MAVLink_statustext_message(2, reason[50:], message_id, 1)
    )
    bridge.poll_status()
    assert bridge.last_statustext == reason.decode("ascii")
    assert capsys.readouterr().out.count("PX4 STATUSTEXT[2]:") == 1


@pytest.mark.parametrize("tail_id,tail_sequence,delay_s", [
    (18, 1, 0.0),  # Another text ID must not extend this message.
    (17, 2, 0.0),  # A missing chunk must not produce a false complete reason.
    (17, 1, 3.0),  # A late tail must not attach to an expired prefix.
])
def test_unrelated_or_incomplete_status_tail_is_discarded(
    tail_id: int, tail_sequence: int, delay_s: float
) -> None:
    """Keep the last useful text when a chunk cannot be safely attached."""
    now = [10.0]
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: now[0])
    bridge._receive_statustext(
        mavlink2.MAVLink_statustext_message(2, b"A" * 50, 17, 0)
    )
    now[0] += delay_s
    bridge._receive_statustext(
        mavlink2.MAVLink_statustext_message(2, b"t", tail_id, tail_sequence)
    )
    assert bridge.last_statustext == "A" * 50


def _stationary_pressure_samples(altitude_m: float) -> np.ndarray:
    """Decodes real HIL packets without stepping physics or opening sockets."""
    wire = io.BytesIO()
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: 10.0)
    bridge.set_local_origin([0.0, 0.0, 0.0])
    bridge.sim_link = SimpleNamespace(mav=mavlink2.MAVLink(wire))
    for _index in range(180):
        bridge.send_state(
            [0.0, 0.0, altitude_m - 100.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 9.80665],
        )
    parser = mavlink2.MAVLink(None)
    messages = parser.parse_buffer(wire.getvalue())
    return np.array(
        [
            message.abs_pressure
            for message in messages
            if message.get_type() == "HIL_SENSOR"
        ]
    )


def test_baro_pressure_tracks_standard_atmosphere() -> None:
    """Use the same 3000 m reference as PX4's test_atmosphere.cpp."""
    pressure_hpa = _stationary_pressure_samples(3000.0)
    assert float(pressure_hpa.mean()) == pytest.approx(701.2, abs=0.15)


def test_stationary_baro_samples_vary_reproducibly() -> None:
    """Fixed pressure with fresh timestamps must not masquerade as a sensor."""
    pressure_hpa = _stationary_pressure_samples(100.0)
    np.testing.assert_array_equal(
        pressure_hpa, _stationary_pressure_samples(100.0)
    )
    pressure_pa = (pressure_hpa * 100.0).astype(np.float32)
    assert np.unique(pressure_pa).size > 100
    assert 0.1 < float(pressure_pa.std()) < 2.0


def test_float_position_precedes_matching_hil_sample() -> None:
    """Meet the existing receiver's pairing contract in either PX4 mode."""
    wire = io.BytesIO()
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: 12345.678)
    bridge.set_local_origin([1.0, 2.0, 0.48])
    bridge.sim_link = SimpleNamespace(
        mav=mavlink2.MAVLink(wire, srcSystem=2, srcComponent=1)
    )
    bridge.last_heartbeat_wall_time = float("inf")
    bridge.send_state(
        [1.001, 1.998, 1.08],
        [0.003, -0.004, 0.005],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 9.80665],
    )
    messages = mavlink2.MAVLink(None).parse_buffer(wire.getvalue())
    assert [message.get_type() for message in messages] == [
        "HIL_SENSOR", "LOCAL_POSITION_NED", "HIL_STATE_QUATERNION"
    ]
    local, hil = messages[1:]
    np.testing.assert_allclose(
        [local.x, local.y, local.z], [0.001, 0.002, -0.6], atol=3e-8
    )
    np.testing.assert_allclose(
        [local.vx, local.vy, local.vz], [0.003, 0.004, -0.005], atol=1e-9
    )
    assert local.time_boot_ms == (hil.time_usec // 1000) & 0xFFFFFFFF
    assert local.get_srcSystem() == hil.get_srcSystem() == 2
    assert local.get_srcComponent() == hil.get_srcComponent() == 1


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_contact_acceleration_saturates_integer_hil_fields_only(
    sign: float,
) -> None:
    """Ground contact must not abort MAVLink packing or clip float sensors."""
    wire = io.BytesIO()
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: 10.0)
    bridge.set_local_origin([0, 0, 0])
    bridge.sim_link = SimpleNamespace(mav=mavlink2.MAVLink(wire))
    bridge.send_state(
        [0, 0, 1], [sign * 400, 0.25, -0.5],
        [sign * 40, -sign * 40, 10], [1, 0, 0, 0], [0, 0, 0],
        [sign * 40, -sign * 40, 9.80665],
    )
    messages = mavlink2.MAVLink(None).parse_buffer(wire.getvalue())
    sensor = next(m for m in messages if m.get_type() == "HIL_SENSOR")
    local = next(m for m in messages if m.get_type() == "LOCAL_POSITION_NED")
    hil = next(m for m in messages if m.get_type() == "HIL_STATE_QUATERNION")
    bound = 32767 if sign > 0 else -32768
    assert [hil.vx, hil.vy, hil.vz] == [bound, -25, 50]
    # Preserve the current receiver's /1000 acceleration convention.
    assert [hil.xacc, hil.yacc, hil.zacc] == [bound, bound, -10000]
    assert local.vx == pytest.approx(sign * 400)
    assert sensor.xacc == pytest.approx(sign * 40)
    assert sensor.yacc == pytest.approx(sign * 40)


def _unlock_test_app(
    now: list[float],
) -> tuple[wind.WindSimulationApp, deque[Any]]:
    """Builds real unlock logic with an in-memory MAVLink transport."""
    messages: deque[Any] = deque()
    bridge = wind.Px4Bridge([1.0] * 4, clock=lambda: now[0])
    bridge.command_link = SimpleNamespace(
        mav=mavlink2.MAVLink(io.BytesIO()),
        recv_match=lambda **_kwargs: messages.popleft() if messages else None,
    )
    app = wind.WindSimulationApp.__new__(wind.WindSimulationApp)
    app.bridge = bridge
    app.clock = lambda: now[0]
    app.flight_control = wind.FlightControlState(np.zeros(3), 1.0)
    app.unlock_requested = False
    app.unlock_request_started_wall_time = None
    app.unlock_ack_accepted = False
    app.last_processed_command_ack = "none"
    app.last_processed_arm_ack_sequence = 0
    app.last_error = "none"
    app.exit_requested = False
    return app, messages


def test_second_identical_arm_rejection_ends_pending_state() -> None:
    """A repeated 400:1 is a new response, even if its text is unchanged."""
    app, messages = _unlock_test_app([10.0])
    for _attempt in range(2):
        app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
        assert app.unlock_requested
        messages.append(mavlink2.MAVLink_command_ack_message(400, 1))
        app.bridge.poll_status()
        app._update_unlock_request_state()
        assert not app.unlock_requested


@pytest.mark.parametrize("ack_result", [None, 0, 5])
def test_unlock_wait_expires_without_confirmed_armed_heartbeat(
    ack_result: int | None,
) -> None:
    """Missing ACK, accepted ACK and in-progress ACK all have a deadline."""
    now = [10.0]
    app, messages = _unlock_test_app(now)
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
    if ack_result is not None:
        messages.append(mavlink2.MAVLink_command_ack_message(400, ack_result))
        app.bridge.poll_status()
    app._update_unlock_request_state()
    assert app.unlock_requested
    now[0] = 16.0
    app._update_unlock_request_state()
    assert not app.unlock_requested
    assert not app.flight_control.armed
    assert "timed out" in app.last_error


def test_repeated_click_does_not_extend_unlock_deadline() -> None:
    """Repeated clicks must not leave a missing reply pending forever."""
    now = [10.0]
    app, _messages = _unlock_test_app(now)
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
    now[0] = 12.0
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
    assert app.unlock_request_started_wall_time == 10.0


def test_accepted_ack_waits_for_confirmed_px4_arm_state() -> None:
    """A successful command ACK alone must not mark the aircraft armed."""
    app, messages = _unlock_test_app([10.0])
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
    messages.append(mavlink2.MAVLink_command_ack_message(400, 0))
    app.bridge.poll_status()
    app._update_unlock_request_state()
    assert app.unlock_requested
    assert not app.flight_control.armed
    assert app.unlock_ack_accepted
    app.flight_control.update_px4_status(armed=True, offboard=False)
    app._update_unlock_request_state()
    assert not app.unlock_requested


def test_emergency_stop_cancels_pending_unlock() -> None:
    """A pending unlock must not keep retrying after the emergency button."""
    app, _messages = _unlock_test_app([10.0])
    app.data = SimpleNamespace(ctrl=np.zeros(4))
    app.bindings = SimpleNamespace(rotor_thrust_actuator_ids=np.arange(4))
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.UNLOCK))
    app._process_ui_command(wind.UiCommand(wind.UiCommandType.EMERGENCY_STOP))
    assert not app.unlock_requested
    assert app.flight_control.flight_state is wind.FlightState.EMERGENCY_STOP


def test_panel_displays_unlock_failure() -> None:
    """The supplied error must be visible in the panel, beyond its color."""
    import queue
    from matplotlib.text import Text

    panel = wind.ControlPanel(queue.SimpleQueue())
    error = "PX4 unlock timed out: no ARM ACK received"
    snapshot = wind.PanelSnapshot(
        px4_connected=True, armed=False, offboard=False,
        flight_state="DISARMED", trajectory="none",
        position_world=np.zeros(3), target_world=np.zeros(3),
        configured_wind_world=np.zeros(3), active_wind_world=np.zeros(3),
        filtered_residual_acceleration=np.zeros(3),
        pinn_force_world=np.zeros(3), wind_enabled=False, last_error=error,
    )
    try:
        panel.update(snapshot)
        visible_text = " ".join(
            artist.get_text() for artist in panel.figure.findobj(Text)
        )
        assert " ".join(error.split()) in " ".join(visible_text.split())
    finally:
        panel.close()


def test_viewer_close_waits_for_render_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not return to interpreter/GLFW teardown while rendering remains."""
    import threading

    close_requested = threading.Event()
    render_exiting = threading.Event()
    allow_render_exit = threading.Event()
    close_returned = threading.Event()

    def renderer(*_args: Any, **kwargs: Any) -> None:
        """Models an asynchronous native viewer exit without opening a GUI."""
        kwargs["handle_return"].put(
            SimpleNamespace(close=close_requested.set)
        )
        close_requested.wait()
        render_exiting.set()
        allow_render_exit.wait()

    monkeypatch.setattr(wind.mujoco.viewer, "_launch_internal", renderer)
    model = wind.mujoco.MjModel.from_xml_string("<mujoco/>")
    viewer = wind.ManagedPassiveViewer(model, wind.mujoco.MjData(model))

    def close_viewer() -> None:
        """Records when the application's close operation actually returns."""
        viewer.close()
        close_returned.set()

    closer = threading.Thread(target=close_viewer)
    closer.start()
    try:
        assert render_exiting.wait(2.0)
        assert not close_returned.is_set()
    finally:
        allow_render_exit.set()
        closer.join(2.0)
    assert close_returned.is_set()
    assert not closer.is_alive()


def test_viewer_startup_failure_reaches_main_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native initialization errors must not strand the startup queue."""
    def renderer(*_args: Any, **_kwargs: Any) -> None:
        """Fails before the viewer handle is available."""
        raise RuntimeError("renderer init failed")

    monkeypatch.setattr(wind.mujoco.viewer, "_launch_internal", renderer)
    model = wind.mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(RuntimeError, match="renderer init failed"):
        wind.ManagedPassiveViewer(model, wind.mujoco.MjData(model))


def test_runtime_closes_viewer_even_when_disarm_transport_fails() -> None:
    """Transport failure must not skip the mandatory render-thread join."""
    closed: list[str] = []

    def fail_disarm() -> None:
        """Models a failed MAVLink write while leaving cleanup possible."""
        raise OSError("disarm transport closed")

    app = wind.WindSimulationApp.__new__(wind.WindSimulationApp)
    app.model = wind.mujoco.MjModel.from_xml_string("<mujoco/>")
    app.data = wind.mujoco.MjData(app.model)
    app.bindings = SimpleNamespace(
        quadrotor_body_id=0,
        rotor_thrust_actuator_ids=np.array([], dtype=int),
    )
    app.arm_actuator_ids = []
    app.arm_hold_controls = np.array([])
    app.exit_requested = False
    app.bridge = SimpleNamespace(
        connected=True, request_disarm=fail_disarm,
        close=lambda: closed.append("bridge"),
    )
    app.viewer = SimpleNamespace(close=lambda: closed.append("viewer"))
    app.panel = SimpleNamespace(close=lambda: closed.append("panel"))
    app.trajectory_plot = SimpleNamespace(close=lambda: closed.append("plot"))
    app.run_logger = None
    with pytest.raises(OSError, match="disarm transport closed"):
        app._close_runtime()
    assert closed == ["viewer", "plot", "panel", "bridge"]
