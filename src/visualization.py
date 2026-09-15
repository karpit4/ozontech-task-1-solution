from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_result(raw_cloud, processed_cloud, obb, gt, prediction, path):
    raw = np.asarray(raw_cloud.points) * 1000.0
    processed = np.asarray(processed_cloud.points) * 1000.0

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Для производительности рисуем подвыборку.
    if len(raw) > 8000:
        idx = np.random.choice(len(raw), 8000, replace=False)
        raw = raw[idx]

    if len(processed) > 8000:
        idx = np.random.choice(len(processed), 8000, replace=False)
        processed = processed[idx]

    ax.scatter(
        raw[:, 0],
        raw[:, 1],
        raw[:, 2],
        s=1,
        alpha=0.10,
        label="raw point cloud",
    )

    ax.scatter(
        processed[:, 0],
        processed[:, 1],
        processed[:, 2],
        s=3,
        alpha=0.7,
        label="processed object",
    )

    # OBB corners.
    corners = np.asarray(obb.get_box_points()) * 1000.0

    edges = [
        (0, 1), (1, 7), (7, 2), (2, 0),
        (3, 6), (6, 5), (5, 4), (4, 3),
        (0, 3), (1, 6), (7, 5), (2, 4),
    ]

    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            linewidth=2,
        )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    ax.set_title(
        "Conveyor dimensioning demo\n"
        f"GT={np.round(gt.dimensions, 1)} mm | "
        f"Measured={np.round(prediction.dimensions, 1)} mm"
    )

    ax.legend()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
