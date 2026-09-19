import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class GroundTruth:
    length_mm: float
    width_mm: float
    height_mm: float
    yaw_deg: float

    @property
    def dimensions(self):
        return np.array(
            [self.length_mm, self.width_mm, self.height_mm],
            dtype=float,
        )


class SyntheticScene:
    """
    Синтетическая сцена:
      z = 0                 -> conveyor
      object sits on z=0
      object is rotated around Z
    """
    SHAPES = ("box", "cylinder", "bottle")
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def _rotation_z(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def make_box(self, dims_mm, yaw_deg=0.0):
        L, W, H = dims_mm

        # Uniform samples on six faces.
        n = max(self.cfg.object_points // 6, 100)

        u = self.rng.random(n)
        v = self.rng.random(n)

        faces = []

        # x = +/- L/2
        for sign in (-1, 1):
            pts = np.column_stack([
                np.full(n, sign * L / 2),
                (u - 0.5) * W,
                v * H,
            ])
            faces.append(pts)

        # y = +/- W/2
        for sign in (-1, 1):
            pts = np.column_stack([
                (u - 0.5) * L,
                np.full(n, sign * W / 2),
                v * H,
            ])
            faces.append(pts)

        # z = 0 / H
        for sign in (0, 1):
            pts = np.column_stack([
                (u - 0.5) * L,
                (v - 0.5) * W,
                np.full(n, sign * H),
            ])
            faces.append(pts)

        points = np.vstack(faces)

        R = self._rotation_z(yaw_deg)
        points = points @ R.T

        # Put the lowest point exactly on the conveyor.
        points[:, 2] -= points[:, 2].min()

        # Depth noise.
        points += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=points.shape,
        )

        return points

    def make_bottle(self, dims_mm, yaw_deg=0.0):
        """
        Коробка + цилиндрический выступ сверху.

        Точки цилиндра лежат только на его поверхности: на боковой стенке
        и на верхней крышке. Нижняя крышка не сэмплируется, потому что она
        прилегает к коробке и снаружи не видна.
        """
        L, W, H = dims_mm

        box_h = H * 0.75
        box = self.make_box((L, W, box_h), yaw_deg)

        radius = min(L, W) * 0.12
        height = H - box_h

        n_total = self.cfg.object_points // 20

        # Делим точки между боковой стенкой и крышкой пропорционально площади,
        # чтобы плотность на обеих частях была одинаковой.
        side_area = 2.0 * np.pi * radius * height
        top_area = np.pi * radius ** 2
        n_side = int(round(n_total * side_area / (side_area + top_area)))
        n_top = n_total - n_side

        # Боковая стенка: равномерно по углу и по высоте.
        theta_side = self.rng.uniform(0, 2 * np.pi, n_side)
        side = np.column_stack([
            radius * np.cos(theta_side),
            radius * np.sin(theta_side),
            box_h + self.rng.random(n_side) * height,
        ])

        # Верхняя крышка: sqrt даёт равномерное распределение по площади диска.
        theta_top = self.rng.uniform(0, 2 * np.pi, n_top)
        r_top = radius * np.sqrt(self.rng.random(n_top))
        top = np.column_stack([
            r_top * np.cos(theta_top),
            r_top * np.sin(theta_top),
            np.full(n_top, H),
        ])

        bump = np.vstack([side, top])

        # Тот же поворот и шум, что и у коробки.
        R = self._rotation_z(yaw_deg)
        bump = bump @ R.T
        bump += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=bump.shape,
        )

        return np.vstack([box, bump])

    def make_cylinder(self, dims_mm, yaw_deg=0.0):
        """
        Вертикальный цилиндр, стоящий на ленте.

        dims_mm = (D, D, H): диаметр берётся из первого элемента,
        высота из третьего. Точки лежат только на поверхности: на боковой
        стенке и на верхней крышке. Нижняя крышка не сэмплируется, так как
        она лежит на ленте и снаружи не видна.
        """
        L, _, H = dims_mm
        radius = L / 2.0

        n_total = self.cfg.object_points

        # Делим точки между стенкой и крышкой пропорционально площади,
        # чтобы плотность была одинаковой.
        side_area = 2.0 * np.pi * radius * H
        top_area = np.pi * radius ** 2
        n_side = int(round(n_total * side_area / (side_area + top_area)))
        n_top = n_total - n_side

        # Боковая стенка: равномерно по углу и по высоте.
        theta_side = self.rng.uniform(0, 2 * np.pi, n_side)
        side = np.column_stack([
            radius * np.cos(theta_side),
            radius * np.sin(theta_side),
            self.rng.random(n_side) * H,
        ])

        # Верхняя крышка: sqrt даёт равномерное распределение по площади диска.
        theta_top = self.rng.uniform(0, 2 * np.pi, n_top)
        r_top = radius * np.sqrt(self.rng.random(n_top))
        top = np.column_stack([
            r_top * np.cos(theta_top),
            r_top * np.sin(theta_top),
            np.full(n_top, H),
        ])

        points = np.vstack([side, top])

        R = self._rotation_z(yaw_deg)
        points = points @ R.T

        points += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=points.shape,
        )

        return points

    def make_conveyor(self):
        x = self.rng.uniform(
            -self.cfg.conveyor_length_mm / 2,
            self.cfg.conveyor_length_mm / 2,
            self.cfg.conveyor_points,
        )
        y = self.rng.uniform(
            -self.cfg.conveyor_width_mm / 2,
            self.cfg.conveyor_width_mm / 2,
            self.cfg.conveyor_points,
        )
        z = self.rng.normal(
            0,
            self.cfg.depth_noise_std_mm * 0.35,
            self.cfg.conveyor_points,
        )
        return np.column_stack([x, y, z])

    def create_scene(self, dims_mm, yaw_deg=0.0, shape = "box"):
        if shape not in self.SHAPES:
            raise ValueError (f"Unknown shape : {shape}. Available : {self.SHAPES}")
        
        if shape == "cylinder":
            dims_mm = (dims_mm[0],dims_mm[0],dims_mm[2])
        
        gt = GroundTruth(
            length_mm=dims_mm[0],
            width_mm=dims_mm[1],
            height_mm=dims_mm[2],
            yaw_deg=yaw_deg,
        )

        if shape=="bottle":
            obj = self.make_bottle(dims_mm, yaw_deg)
        elif shape == "box":
            obj = self.make_box(dims_mm, yaw_deg)
        elif shape == "cylinder":
            obj = self.make_cylinder(dims_mm, yaw_deg)

        conveyor = self.make_conveyor()

        # Random depth outliers.
        outliers = np.column_stack([
            self.rng.uniform(-600, 600, self.cfg.outlier_points),
            self.rng.uniform(-300, 300, self.cfg.outlier_points),
            self.rng.uniform(0, 350, self.cfg.outlier_points),
        ])

        cloud = np.vstack([conveyor, obj, outliers])

        return o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cloud / 1000.0)
        ), gt
