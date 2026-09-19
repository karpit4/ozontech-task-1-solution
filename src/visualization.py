from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _set_equal_aspect(ax, points):
    """Set approximately equal scale on X/Y/Z axes."""

    ax.set_box_aspect((1,1,1))

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0

    # Small margin around the scene.
    half_range *= 1.10

    ax.set_xlim(
        center[0] - half_range,
        center[0] + half_range,
    )
    ax.set_ylim(
        center[1] - half_range,
        center[1] + half_range,
    )
    ax.set_zlim(
        center[2] - half_range,
        center[2] + half_range,
    )

def _draw_obb(ax, obb):
    """Draw oriented bounding box."""

    corners = np.asarray(obb.get_box_points()) * 1000.0

    edges = [
        (0, 1), (1, 7), (7, 2), (2, 0),  # передняя грань
        (3, 6), (6, 4), (4, 5), (5, 3),  # задняя грань
        (0, 3), (1, 6), (2, 5), (7, 4),  # соединяющие рёбра
    ]

    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            linewidth=2.5,
            label="_nolegend_",
        )

    ax.scatter(
        corners[:, 0],
        corners[:, 1],
        corners[:, 2],
        s=20,
        label="OBB corners",
    )

    return corners

PASS_COLOR = "#2e7d32"
FAIL_COLOR = "#c62828"

# Ширина одного символа моноширинного шрифта в долях размера шрифта
# (DejaVu Sans Mono, шрифт matplotlib по умолчанию для family="monospace").
_CHAR_W = 0.602
_ROW_CHARS = 40  # длина строки таблицы в символах


def _draw_report(ax, metrics):
    """Draw the measurement report as a text panel (no axes, no frame)."""

    ax.axis("off")

    size = 10
    # Смещение статуса от начала строки: конец строки таблицы + небольшой зазор.
    status_dx = _ROW_CHARS * _CHAR_W * size + 14

    def put(y, s, color="black", size=size, weight="normal", dx=0.0):
        ax.annotate(
            s,
            xy=(0.0, y),
            xycoords="axes fraction",
            xytext=(dx, 0),
            textcoords="offset points",
            family="monospace",
            fontsize=size,
            fontweight=weight,
            color=color,
            ha="left",
            va="center",
        )

    def status(ok):
        return ("PASS", PASS_COLOR) if ok else ("FAIL", FAIL_COLOR)

    y = 0.62
    put(y, "=== MEASUREMENT RESULT ===", weight="bold")

    y -= 0.07
    put(
        y,
        f"{'[mm]':<5}{'GT':>9}{'pred':>9}{'error':>9}{'tol':>8}",
        color="dimgray",
    )

    for i, name in enumerate(("L", "W", "H")):
        y -= 0.06
        tol = f"±{metrics['tolerance_mm'][i]:.2f}"
        put(
            y,
            f"{name:<5}"
            f"{metrics['gt_mm'][i]:>9.2f}"
            f"{metrics['pred_mm'][i]:>9.2f}"
            f"{metrics['error_mm'][i]:>+9.2f}"
            f"{tol:>8}",
        )
        text, color = status(metrics["within_tolerance"][i])
        put(y, text, color=color, weight="bold", dx=status_dx)

    y -= 0.10
    text, color = status(metrics["pass"])
    put(y, f"OBJECT: {text}", color=color, size=14, weight="bold")

def plot_result(
    raw_cloud,
    processed_cloud,
    obb,
    metrics,
    path,
    seed,
    show=True,
):
    """
    Visualize conveyor point cloud, processed object and MOBB.
    The measurement report is drawn as a text panel next to the 3D plot.

    Parameters
    ----------
    raw_cloud:
        Original point cloud.

    processed_cloud:
        Point cloud after segmentation/filtering.

    obb:
        Open3D oriented bounding box.

    metrics:
        Result of evaluation.evaluate(gt, prediction).

    path:
        Path where PNG image (plot + report) will be saved.

    seed:
        Seed for the visualization-only random downsampling.

    show:
        If True, display an interactive Matplotlib window.
    """

    # Convert meters -> millimeters.
    raw = np.asarray(raw_cloud.points) * 1000.0
    processed = np.asarray(processed_cloud.points) * 1000.0

    # Downsample only for visualization.
    # The actual measurement is NOT affected.
    rng = np.random.default_rng(seed)

    if len(raw) > 8000:
        idx = rng.choice(len(raw), 8000, replace=False)
        raw = raw[idx]

    if len(processed) > 8000:
        idx = rng.choice(len(processed), 8000, replace=False)
        processed = processed[idx]

    # Figure: 3D plot on the left, text report on the right.
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.5])

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_report = fig.add_subplot(gs[0, 1])

    # Raw point cloud.
    if len(raw) > 0:
        ax.scatter(
            raw[:, 0],
            raw[:, 1],
            raw[:, 2],
            s=1,
            alpha=0.12,
            label="raw point cloud",
        )

    # Processed object.
    if len(processed) > 0:
        ax.scatter(
            processed[:, 0],
            processed[:, 1],
            processed[:, 2],
            s=4,
            alpha=0.3,
            label="processed object",
        )

    # MOBB.
    obb_corners = _draw_obb(ax, obb)

    # Include both point cloud and MOBB when calculating
    # visualization limits.
    all_points = np.vstack([raw, processed, obb_corners])
    _set_equal_aspect(ax, all_points)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Conveyor Dimensioning Demo")
    ax.legend()

    # Measurement report.
    _draw_report(ax_report, metrics)

    # Save PNG.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")

    # Interactive window.
    if show:
        plt.show()
    else:
        plt.close(fig)