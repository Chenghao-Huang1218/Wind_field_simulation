#!/usr/bin/env python3
"""Plot measured outputs of the offline reference check, never flight data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_figure(report: dict[str, Any]) -> Figure:
    """Show ordered speed/direction responses of the nominal static reference.

    Args:
        report: JSON output from check_x4_950_reference.py.

    Returns:
        Two-panel figure. Every marker is a deterministic reference evaluation;
        lines connect ordered conditions, with no inferred statistical error.
    """
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    grid = [row for row in report["rest_wind_grid"]
            if row["state"] == "rest_identity"]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=100,
                                constrained_layout=True)
    for direction, color, marker, style in zip(
        (0, 45, 90), ("#0072B2", "#D55E00", "#009E73"),
        ("o", "s", "^"), ("-", "--", ":"), strict=True,
    ):
        rows = [row for row in grid
                if row["wind_direction_deg"] == direction]
        speed = [row["wind_speed_m_s"] for row in rows]
        acceleration = [np.linalg.norm(row["increment_world_m_s2"])
                        for row in rows]
        axes[0].plot(speed, acceleration, color=color, marker=marker,
                     linestyle=style, markersize=3, linewidth=1.2,
                     label=f"{direction}°")
    rows = [row for row in grid if row["wind_speed_m_s"] == 3]
    directions = [row["wind_direction_deg"] for row in rows]
    response = [np.linalg.norm(row["increment_world_m_s2"]) for row in rows]
    axes[1].plot(directions + [360], response + [response[0]],
                 color="#0072B2", marker="o", markersize=3, linewidth=1.2)
    axes[0].set(xlabel="Horizontal wind speed (m/s)",
                title="(a) Fixed wind directions", xlim=(0, 3))
    axes[1].set(xlabel="Wind direction in world XY (deg)",
                title="(b) Direction scan at 3 m/s", xlim=(0, 360))
    axes[0].legend(title="Wind direction", frameon=False, loc="upper left")
    axes[1].set_xticks([0, 90, 180, 270, 360])
    for axis in axes:
        axis.set_ylabel("Horizontal increment norm (m/s²)")
        axis.set_ylim(0, 0.145)
        axis.grid(alpha=0.18, linewidth=0.5)
    return figure


def main() -> int:
    """Read the reference check and export PNG/PDF/SVG at fixed dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "outputs/pinn/x4_950_wind_v1/reference_check.json",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "docs/figures")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    figure = create_figure(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.output_dir / f"x4-950-reference-v1.{extension}",
                       dpi=300)
    plt.close(figure)
    print(f"Reference behavior figure: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
