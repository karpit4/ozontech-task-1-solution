# Conveyor Dimensioning Demo

демо алгоритма:

`RGB-D point cloud -> conveyor plane removal -> outlier filtering -> convex hull -> minimum-volume OBB -> L/W/H -> quality check`

## Что умеет demo

1. Генерирует синтетические 3D-объекты произвольной формы.
2. Имитирует шум depth-сенсора и движущийся конвейер.
3. Удаляет плоскость конвейера через RANSAC.
4. Очищает облако точек.
5. Строит convex hull.
6. Вычисляет минимальный OBB.
7. Сравнивает результат с ground truth.
8. Проверяет допуск `max(5%, 5 mm)`.
9. Строит визуализацию pipeline.

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

```bash
python -m src.demo
```

После запуска результат появится в `output/`.
