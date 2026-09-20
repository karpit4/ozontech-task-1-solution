# Object dimension measurement on a conveyor demo

[Русская версия](README.ru.md)

Demo of a conveyor-based object dimensioning algorithm:

`RGB-D point cloud -> conveyor plane removal -> outlier filtering -> object extraction -> convex hull -> minimum-volume OBB -> L/W/H -> quality check`

## What the demo does

1. Generates synthetic 3D objects: `box`, `cylinder`, `bottle`, `sphere`, `pencil`.
2. Simulates depth sensor noise and a moving conveyor.
3. Removes the conveyor plane using RANSAC.
4. Removes outliers using radius and statistical filtering.
5. Extracts the main object component.
6. Builds a convex hull.
7. Computes a minimum-volume OBB.
8. Compares the result with ground truth.
9. Checks the tolerance `max(5%, 5 mm)`.
10. Provides a benchmark for objects of different sizes and shapes.

## Project structure

```text
src/
├── demo.py            — demo entry point
├── scene.py           — synthetic scene and point cloud generation
├── objects.py         — 3D object generation
├── pipeline.py        — main dimensioning algorithm
├── evaluation.py      — measurement accuracy evaluation
├── visualization.py   — point cloud and OBB visualization
├── benchmark.py       — performance and accuracy benchmark
├── config.py          — algorithm parameters
└── __init__.py        — defines src as a Python package
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

## Running

Run the demo:

```bash
python -m src.demo
```

Run the benchmark:

```bash
python -m src.benchmark
```

After running the demo, the visualization is saved to:

```text
output/dimensioning_result.png
<<<<<<< HEAD
```
=======
```
>>>>>>> ab96f86d8d25ab28dca56a5abe2b86886257833a
