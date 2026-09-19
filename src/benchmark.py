import time

import numpy as np

from .config import Config
from .scene import SyntheticScene
from .pipeline import DimensioningPipeline
from .evaluation import evaluate


def fmt(values, spec=".1f"):
    """Три числа -> '(409.7, 309.4, 299.9)' (обычные float, без np.float64)."""
    return "(" + ", ".join(format(float(v), spec) for v in values) + ")"

def main():
    cfg = Config()
    scene = SyntheticScene(cfg)
    pipeline = DimensioningPipeline(cfg)

    # Отдельный генератор для углов поворота. Сид сдвинут на 1, чтобы поток
    # случайных чисел не совпадал с потоком, который использует сцена.
    yaw_rng = np.random.default_rng(cfg.seed + 1)

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
        f"{'GT [mm]':<17}"
        f"{'Measured [mm]':<23}"
        f"{'Error [mm]':<25}"
        f"{'Time [ms]':<10}"
        f"{'Result':<8}"
    )

    for dims in test_cases:
        cloud, gt = scene.create_scene(
            dims_mm=dims,
            yaw_deg=yaw_rng.uniform(0, 90),
            shape="box",
        )

        t0 = time.perf_counter()
        result = pipeline.measure(cloud)
        elapsed = (time.perf_counter() - t0) * 1000

        metrics = evaluate(gt, result["measurement"])
        times.append(elapsed)

        print(
            f"{fmt(metrics['gt_mm'], 'g'):<17}"
            f"{fmt(metrics['pred_mm']):<23}"
            f"{fmt(metrics['error_mm'], '+.1f'):<25}"
            f"{elapsed:<10.1f}"
            f"{'PASS' if metrics['pass'] else 'FAIL':<8}"
        )

    print("\nTiming:")
    print(f"P50: {np.percentile(times, 50):.1f} ms")
    print(f"P95: {np.percentile(times, 95):.1f} ms")
    print(f"Max: {np.max(times):.1f} ms")


if __name__ == "__main__":
    main()