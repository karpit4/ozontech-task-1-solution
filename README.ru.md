# Демо измерения габаритов объекта на конвейере

[English version](README.md)

Демо алгоритма измерения габаритов объектов на конвейере:

`RGB-D point cloud -> удаление плоскости конвейера -> фильтрация выбросов -> выделение объекта -> convex hull -> minimum-volume OBB -> L/W/H -> проверка качества`

## Что умеет demo

1. Генерирует синтетические 3D-объекты: `box`, `cylinder`, `bottle`, `sphere`, `pencil`.
2. Имитирует шум depth-сенсора и движущийся конвейер.
3. Удаляет плоскость конвейера через RANSAC.
4. Удаляет выбросы с помощью radius и statistical filtering.
5. Выделяет основной компонент объекта.
6. Строит convex hull.
7. Вычисляет минимальный OBB.
8. Сравнивает результат с ground truth.
9. Проверяет допуск `max(5%, 5 mm)`.
10. Позволяет проводить benchmark для объектов разных размеров и форм.

## Структура проекта

```text
src/
├── demo.py            — точка входа, запуск демонстрации
├── scene.py           — генерация синтетической сцены и point cloud
├── objects.py         — генерация 3D-объектов
├── pipeline.py        — основной алгоритм измерения
├── evaluation.py      — оценка точности измерений
├── visualization.py   — визуализация point cloud и OBB
├── benchmark.py       — benchmark производительности и точности
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

Запуск демонстрации:

```bash
python -m src.demo
```

Запуск benchmark:

```bash
python -m src.benchmark
```

После запуска демонстрации визуализация сохраняется в:

```text
output/dimensioning_result.png
```
