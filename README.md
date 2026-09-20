# Conveyor Dimensioning Demo

демо алгоритма:

`RGB-D point cloud -> conveyor plane removal -> outlier filtering -> convex hull -> minimum-volume OBB -> L/W/H -> quality check`

## Что умеет demo

1. Генерирует синтетические 3D-объекты: `box`, `cylinder`, `bottle`, `sphere`, `pencil`.
2. Имитирует шум depth-сенсора и конвейер с выбросами.
3. Удаляет плоскость конвейера через RANSAC.
4. Очищает облако точек с помощью voxel, radius и statistical filtering.
5. Строит convex hull.
6. Вычисляет минимальный OBB.
7. Сравнивает результат с ground truth.
8. Проверяет допуск `max(5%, 5 mm)`.
9. Строит визуализацию pipeline.
10. Позволяет запускать benchmark для объектов разных размеров и форм.

## Структура проекта

```text
src/
├── demo.py            — точка входа, запуск демонстрации
├── scene.py           — генерация синтетической сцены и point cloud
├── objects.py         — генерация 3D-объектов
├── pipeline.py        — основной алгоритм измерения
├── evaluation.py      — оценка точности измерений
├── visualization.py   — визуализация point cloud и OBB
├── benchmark.py       — тестирование производительности
├── config.py          — параметры алгоритма
└── __init__.py        — определяет src как Python-пакет
```

## Установка

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Запуск

Демонстрация:

```bash
python -m src.demo
```

Benchmark:

```bash
python -m src.benchmark
```

После запуска demo результат визуализации появится в `output/dimensioning_result.png`.
