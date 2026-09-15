import numpy as np


def tolerance_mm(gt_mm):
    return np.maximum(0.05 * gt_mm, 5.0)


def evaluate(gt, prediction):
    gt_dims = np.asarray(gt.dimensions, dtype=float)
    pred_dims = np.asarray(prediction.dimensions, dtype=float)

    # Для демонстрации считаем L >= W >= H.
    gt_dims = np.sort(gt_dims)[::-1]
    pred_dims = np.sort(pred_dims)[::-1]

    error = pred_dims - gt_dims
    abs_error = np.abs(error)
    tol = tolerance_mm(gt_dims)

    return {
        "gt_mm": gt_dims,
        "pred_mm": pred_dims,
        "error_mm": error,
        "abs_error_mm": abs_error,
        "tolerance_mm": tol,
        "within_tolerance": abs_error <= tol,
        "pass": bool(np.all(abs_error <= tol)),
    }


def print_report(result):
    print("\n=== MEASUREMENT RESULT ===")

    names = ["L", "W", "H"]

    for i, name in enumerate(names):
        print(
            f"{name}: "
            f"GT={result['gt_mm'][i]:7.2f} mm | "
            f"pred={result['pred_mm'][i]:7.2f} mm | "
            f"error={result['error_mm'][i]:+7.2f} mm | "
            f"tol=±{result['tolerance_mm'][i]:5.2f} mm | "
            f"{'PASS' if result['within_tolerance'][i] else 'FAIL'}"
        )

    print(f"\nOBJECT: {'PASS' if result['pass'] else 'FAIL'}")
