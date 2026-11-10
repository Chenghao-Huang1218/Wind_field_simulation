#!/usr/bin/env python3
"""Plot one pilot's learning curve and raw diagnostic holdout predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def create_figure(pilot_dir: Path) -> Figure:
    """Show convergence, XY parity and error versus wind without uncertainty claims.

    Args:
        pilot_dir: Run root with dataset, training_corrected, evaluation_corrected.

    Returns:
        Fixed-size three-panel figure; 500 heldout pairs, 50 independent sampling
        groups. All scatter points shown; no CI, smoothing or extra fitting.
    """
    history = json.loads((pilot_dir / "training_corrected/history.json").read_text())
    with np.load(pilot_dir / "evaluation_corrected/predictions.npz") as stored:
        heldout = stored["split"] == 2
        prediction = stored["prediction"][heldout]
        label = stored["label"][heldout]
    with np.load(pilot_dir / "dataset/pairs.npz") as stored:
        wind_speed = np.linalg.norm(stored["wind_world"][heldout, :2], axis=1)
    error = np.linalg.norm(prediction[:, :2] - label[:, :2], axis=1)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), dpi=100, constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].semilogy(epochs, [row["training_loss"] for row in history],
                     color="#0072B2", label="Train", linewidth=1.2)
    axes[0].semilogy(epochs, [row["validation_loss"] for row in history],
                     color="#D55E00", label="Validation", linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="Epoch", ylabel="Combined loss (log scale)", title="(a) One seed-42 pilot")
    axes[0].legend(frameon=False, loc="upper right")
    limits = [min(label[:, :2].min(), prediction[:, :2].min()) - .02,
              max(label[:, :2].max(), prediction[:, :2].max()) + .02]
    axes[1].plot(limits, limits, color=".5", linewidth=.8, linestyle="--")
    for axis, color, marker in ((0, "#0072B2", "o"), (1, "#D55E00", "+")):
        axes[1].scatter(label[:, axis], prediction[:, axis], s=8, color=color,
                         marker=marker, alpha=.65, linewidths=.5, label="XY"[axis])
    axes[1].set(xlabel="Reference increment (m/s²)", ylabel="PINN increment (m/s²)",
                 title="(b) Pilot holdout: 500 pairs", xlim=limits, ylim=limits)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend(frameon=False, loc="upper left")
    axes[2].scatter(wind_speed, 1000*error, s=8, alpha=.55, color="#0072B2", linewidths=0)
    axes[2].set(xlabel="Horizontal wind speed (m/s)",
                 ylabel="XY vector error (10⁻³ m/s²)",
                 title="(c) Raw error; no EMA/clip", xlim=(-.05, 3.05), ylim=(0, None))
    for axis in axes:
        axis.set_box_aspect(1)
        axis.grid(alpha=.15, linewidth=.5)
    with (pilot_dir / "evaluation_corrected/plot_data.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wind_speed_m_s", "reference_x_m_s2", "reference_y_m_s2",
                         "prediction_x_m_s2", "prediction_y_m_s2", "vector_error_m_s2"])
        writer.writerows(np.column_stack((wind_speed, label[:, :2], prediction[:, :2], error)))
    return figure


def main() -> None:
    """Export diagnostic PNG/PDF/SVG; existing simulation assets are untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    figure = create_figure(args.pilot_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.output_dir / f"x4-950-pinn-pilot-v1.{extension}", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
