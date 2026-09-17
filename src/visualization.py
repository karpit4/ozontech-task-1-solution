from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _set_equal_aspect(ax, points):
    """Set approximately equal scale on X/Y/Z axes."""

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


def plot_result(
    raw_cloud,
    processed_cloud,
    obb,
    gt,
    prediction,
    path,
    show=True,
):
    """
    Visualize conveyor point cloud, processed object and MOBB.

    Parameters
    ----------
    raw_cloud:
        Original point cloud.

    processed_cloud:
        Point cloud after segmentation/filtering.

    obb:
        Open3D oriented bounding box.

    gt:
        Ground-truth object dimensions.

    prediction:
        Measured dimensions.

    path:
        Path where PNG image will be saved.

    show:
        If True, display an interactive Matplotlib 3D window.
    """

    # ---------------------------------------------------------
    # Convert meters -> millimeters.
    # ---------------------------------------------------------

    raw = np.asarray(raw_cloud.points) * 1000.0
    processed = np.asarray(processed_cloud.points) * 1000.0

    # ---------------------------------------------------------
    # Downsample only for visualization.
    # The actual measurement is NOT affected.
    # ---------------------------------------------------------

    rng = np.random.default_rng(42)

    if len(raw) > 8000:
        idx = rng.choice(
            len(raw),
            8000,
            replace=False,
        )
        raw = raw[idx]

    if len(processed) > 8000:
        idx = rng.choice(
            len(processed),
            8000,
            replace=False,
        )
        processed = processed[idx]

    # ---------------------------------------------------------
    # Create figure.
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------------------------------------
    # Raw point cloud.
    # ---------------------------------------------------------

    if len(raw) > 0:
        ax.scatter(
            raw[:, 0],
            raw[:, 1],
            raw[:, 2],
            s=1,
            alpha=0.12,
            label="raw point cloud",
        )

    # ---------------------------------------------------------
    # Processed object.
    # ---------------------------------------------------------

    if len(processed) > 0:
        ax.scatter(
            processed[:, 0],
            processed[:, 1],
            processed[:, 2],
            s=4,
            alpha=0.3,
            label="processed object",
        )

    # ---------------------------------------------------------
    # MOBB.
    # ---------------------------------------------------------

    obb_corners = _draw_obb(ax, obb)

    # Include both point cloud and MOBB when calculating
    # visualization limits.
    all_points = np.vstack(
        [
            raw,
            processed,
            obb_corners,
        ]
    )

    _set_equal_aspect(ax, all_points)

    # ---------------------------------------------------------
    # Axes.
    # ---------------------------------------------------------

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # ---------------------------------------------------------
    # Title.
    # ---------------------------------------------------------

    gt_dims = np.round(gt.dimensions, 1)
    measured_dims = np.round(prediction.dimensions, 1)

    error = np.abs(
        measured_dims - gt_dims
    )

    ax.set_title(
        "Conveyor Dimensioning Demo\n"
        f"GT: {gt_dims} mm    |    "
        f"Measured: {measured_dims} mm    |    "
        f"Error: {np.round(error, 1)} mm"
    )

    ax.legend()

    # ---------------------------------------------------------
    # Save PNG.
    # ---------------------------------------------------------

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    # ---------------------------------------------------------
    # Interactive window.
    # ---------------------------------------------------------

    if show:
        plt.show()

    else:
        plt.close(fig)