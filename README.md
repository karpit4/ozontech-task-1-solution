# Conveyor Dimensioning Demo

[Русская версия](README_ru.md)

Demo of an algorithm that measures object dimensions on a conveyor:

`RGB-D point cloud -> conveyor plane removal -> outlier filtering -> convex hull -> minimum-volume OBB -> size corrections -> L/W/H -> quality check`

## What the demo does

1. Generates synthetic 3D objects: `box`, `cylinder`, `bottle`, `sphere`, `pencil`.
2. Simulates depth-sensor noise, a conveyor belt and random outliers.
3. Removes the conveyor plane with RANSAC.
4. Removes outliers with an adaptive radius filter and statistical filtering (SOR).
5. Builds the convex hull and the minimum-volume OBB.
6. Applies constant corrections to the measured height, length and width.
7. Compares the result with the ground truth.
8. Checks the tolerance `max(5%, 5 mm)` for each dimension.
9. Draws the point cloud, the OBB and a measurement report.
10. Runs a benchmark over objects of different sizes and shapes.

## Project structure

```text
src/
├── demo.py            — entry point, runs the demonstration
├── scene.py           — synthetic scene: conveyor, object, outliers
├── objects.py         — 3D object generators
├── pipeline.py        — measurement algorithm
├── evaluation.py      — accuracy evaluation
├── visualization.py   — point cloud, OBB and report plot
├── benchmark.py       — accuracy and speed benchmark
├── config.py          — parameters
└── __init__.py        — marks src as a Python package
```

## Installation

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Run the demo:

```bash
python -m src.demo
```

The visualization with the measurement report is saved to:

```text
output/dimensioning_result.png
```

Run the benchmark:

```bash
python -m src.benchmark
```

For every shape the benchmark measures `N_SIZES` cubes from 10 to 400 mm. It prints errors per object and per shape, timing (P50 / P95 / max) and the number of passed objects.

## Configuration

All parameters are in `src/config.py`:

- object for the demo: `object_shape`, `object_dims`, `object_yaw`;
- sensor model: `depth_noise_std_mm`, `outlier_points`;
- filtering: voxel size, RANSAC, radius filter and SOR parameters;
- corrections: `height_correction_mm`, `length_correction_mm`, `width_correction_mm` (added to the measured value, a negative number reduces it);
- `seed` makes the generated scene reproducible.

## Notes

- The scene is synthetic. The correction constants were tuned experimentally on it and have to be re-tuned for a real sensor.
- The point cloud is the same on every run, but the Open3D steps (RANSAC, OBB) can give slightly different measurements between runs.