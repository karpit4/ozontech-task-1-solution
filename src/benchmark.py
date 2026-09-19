import time

import numpy as np

from .config import Config
from .scene import SyntheticScene
from .pipeline import DimensioningPipeline
from .evaluation import evaluate


def main():
    cfg = Config()
    scene = SyntheticScene(cfg)
    pipeline = DimensioningPipeline(cfg)

    test_cases = [
        (10, 10, 10),
        (20, 20, 20),
        (50, 50, 50),
        (100, 50, 20),
        (200, 100, 10),
        (400, 300, 300),
    ]

    times = []

    print(
        f"{'GT [mm]':<22}"
        f"{'Measured [mm]':<30}"
        f"{'Time [ms]':<12}"
        f"{'PASS':<8}"
    )

    for dims in test_cases:
        cloud, gt = scene.create_scene(
            dims_mm=dims,
            yaw_deg=np.random.uniform(0, 90),
            shape="box",
        )

        t0 = time.perf_counter()
        result = pipeline.measure(cloud)
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = evaluate(gt, result["measurement"])
        times.append(elapsed)

        print(
            f"{str(tuple(dims)):<22}"
            f"{str(tuple(np.round(metrics['pred_mm'], 1))):<30}"
            f"{elapsed:<12.1f}"
            f"{str(metrics['pass']):<8}"
        )

    print("\nTiming:")
    print(f"P50: {np.percentile(times, 50):.1f} ms")
    print(f"P95: {np.percentile(times, 95):.1f} ms")
    print(f"Max: {np.max(times):.1f} ms")


if __name__ == "__main__":
    main()
