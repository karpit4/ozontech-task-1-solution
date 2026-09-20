import time

import numpy as np

from .config import Config
from .scene import SyntheticScene
from .pipeline import DimensioningPipeline
from .evaluation import evaluate
from .objects import ObjectGenerator

# --- Benchmark parameters ---------------------------------------------------
# N_SIZES - number of object of one shape
N_SIZES = 30
MIN_SIZE_MM = 10.0
MAX_SIZE_MM = 400.0
MAX_YAW_DEG = 90.0


def fmt(values, spec=".1f"):
    """Три числа -> '(409.7, 309.4, 299.9)' (обычные float, без np.float64)."""
    return "(" + ", ".join(format(float(v), spec) for v in values) + ")"


def plural(word):
    """'box' -> 'boxes', 'cylinder' -> 'cylinders', 'ability' -> 'abilities'."""
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def make_sizes():
    """
    Кубы со стороной от MIN_SIZE_MM до MAX_SIZE_MM с равным шагом.
    Возвращает список (L, W, H) в мм.
    """
    sides = np.round(np.linspace(MIN_SIZE_MM, MAX_SIZE_MM, N_SIZES))
    return [(float(s), float(s), float(s)) for s in sides]


def main():
    cfg = Config()
    scene = SyntheticScene(cfg)
    pipeline = DimensioningPipeline(cfg)
    objects = ObjectGenerator(cfg)

    shapes = list(objects.generators.keys())
    sizes = make_sizes()

    # Углы поворота берём из отдельного генератора (сид сдвинут на 1, чтобы
    # его поток не совпадал с потоком сцены) и заранее, по одному на размер:
    # так они не зависят от числа форм, и i-й объект каждой формы
    # повёрнут одинаково.
    yaw_rng = np.random.default_rng(cfg.seed + 1)
    yaws = yaw_rng.uniform(0, MAX_YAW_DEG, size=len(sizes))

    # Прогрев: первый вызов обычно медленнее остальных, не учитываем его во
    # времени. Отдельная сцена, чтобы не трогать генератор основной.
    warm_cloud, _ = SyntheticScene(cfg).create_scene(
        dims_mm=sizes[len(sizes) // 2],
        yaw_deg=0.0,
        shape=shapes[0],
    )
    pipeline.measure(warm_cloud)

    times = []
    passed = {shape: 0 for shape in shapes}
    # Знаковые ошибки (pred - GT) по каждой форме: три числа на объект.
    errors = {shape: [] for shape in shapes}

    print(
        f"{'GT [mm]':<17}"
        f"{'Measured [mm]':<23}"
        f"{'Error [mm]':<25}"
        f"{'Time [ms]':<10}"
        f"{'Result':<8}"
    )

    for shape in shapes:
        print(f"\n--- {shape} ---")

        for dims, yaw in zip(sizes, yaws):
            cloud, gt = scene.create_scene(
                dims_mm=dims,
                yaw_deg=yaw,
                shape=shape,
            )

            t0 = time.perf_counter()
            try:
                result = pipeline.measure(cloud)
            except Exception as exc:
                # Сбой на одном объекте не должен ронять весь бенчмарк:
                # считаем объект непройденным и идём дальше.
                print(f"{fmt(gt.dimensions, 'g'):<17}ERROR: {exc}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000

            metrics = evaluate(gt, result["measurement"])
            times.append(elapsed)
            passed[shape] += int(metrics["pass"])
            errors[shape].append(metrics["error_mm"])

            print(
                f"{fmt(metrics['gt_mm'], 'g'):<17}"
                f"{fmt(metrics['pred_mm']):<23}"
                f"{fmt(metrics['error_mm'], '+.1f'):<25}"
                f"{elapsed:<10.1f}"
                f"{'PASS' if metrics['pass'] else 'FAIL':<8}"
            )

        # Средняя ошибка по форме, под колонкой Error.
        if errors[shape]:
            mean_error = np.mean(errors[shape], axis=0)
            print(f"{'Mean error':<40}{fmt(mean_error, '+.1f')}")

    if times:
        print("\nTiming:")
        print(f"P50: {np.percentile(times, 50):.1f} ms")
        print(f"P95: {np.percentile(times, 95):.1f} ms")
        print(f"Max: {np.max(times):.1f} ms")

    # --- Итоговая статистика ---------------------------------------------
    total = len(sizes) * len(shapes)
    total_passed = sum(passed.values())
    all_errors = [e for shape in shapes for e in errors[shape]]

    print(f"\nTotal objects: {total}")
    print(f"Passed objects: {total_passed}")
    print(f"Pass rate: {total_passed / total:.2f}")

    if all_errors:
        mean_all = np.mean(all_errors, axis=0)
        print(f"mean error : {fmt(mean_all, '+.1f')}")

    labels = {shape: f"{plural(shape).capitalize()} passed" for shape in shapes}
    width = max(len(label) for label in labels.values())

    for shape in shapes:
        print(f"{labels[shape]:<{width}} : {passed[shape]} / {len(sizes)}")


if __name__ == "__main__":
    main()