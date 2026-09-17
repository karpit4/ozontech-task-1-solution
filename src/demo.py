from pathlib import Path

import numpy as np

from .config import Config
from .scene import SyntheticScene
from .pipeline import DimensioningPipeline
from .evaluation import evaluate, print_report
from .visualization import plot_result

OBJECT_DIMS = (400, 300, 150)
OBJECT_YAW = 90



def main():
    np.random.seed(42)

    cfg = Config()
    scene = SyntheticScene(cfg)
    pipeline = DimensioningPipeline(cfg)


    cloud, gt = scene.create_scene(
        dims_mm=OBJECT_DIMS,
        yaw_deg=OBJECT_YAW,
        irregular=True,
    )

    result = pipeline.measure(cloud)
    measurement = result["measurement"]

    metrics = evaluate(gt, measurement)
    print_report(metrics)

    print("\n=== PIPELINE ===")
    print(f"Raw points:       {len(cloud.points)}")
    print(f"After background: {len(result['cloud_after_plane'].points)}")
    print(f"After filtering:  {len(result['cloud_filtered'].points)}")
    print(f"Plane model:      {np.round(result['plane_model'], 6)}")

    if cfg.save_visualization:
        out = Path("output")
        out.mkdir(exist_ok=True)

        plot_result(
            raw_cloud=cloud,
            processed_cloud=result["cloud_filtered"],
            obb=result["obb"],
            gt=gt,
            prediction=measurement,
            path=out / "dimensioning_result.png",
        )

        print(f"\nVisualization: {out / 'dimensioning_result.png'}")


if __name__ == "__main__":
    main()
