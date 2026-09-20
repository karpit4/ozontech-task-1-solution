import numpy as np
import math 

class ObjectGenerator:
    def __init__(self,cfg):
        self.cfg = cfg 
        self.rng = np.random.default_rng(cfg.seed)
        self.generators = {
            "box": self.make_box,
            "cylinder": self.make_cylinder,
            "bottle": self.make_bottle,
            "sphere": self.make_sphere,
            "pencil": self.make_pencil,
        }
        
        
    @staticmethod
    def _rotation_z(deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    
    def create(self, shape, dims_mm, yaw_deg=0.0):
        if shape not in self.generators:
            raise ValueError(
                f"Unknown shape: {shape}. "
                f"Available: {tuple(self.generators)}"
            )

        return self.generators[shape](dims_mm, yaw_deg)

    def _make_box_points(self, dims_mm, n_points):
        L, W, H = dims_mm

        # Number of points per face.
        n = max(n_points // 6, 100)

        # Shared random coordinates.
        u = self.rng.random(n)
        v = self.rng.random(n)

        faces = []

        # x = +/- L/2
        for sign in (-1, 1):
            pts = np.column_stack([
                np.full(n, sign * L / 2),
                (u - 0.5) * W,
                (v - 0.5) * H,
            ])
            faces.append(pts)

        # y = +/- W/2
        for sign in (-1, 1):
            pts = np.column_stack([
                (u - 0.5) * L,
                np.full(n, sign * W / 2),
                (v - 0.5) * H,
            ])
            faces.append(pts)

        # z = +/- H/2
        for sign in (-1, 1):
            pts = np.column_stack([
                (u - 0.5) * L,
                (v - 0.5) * W,
                np.full(n, sign * H / 2),
            ])
            faces.append(pts)

        return np.vstack(faces)

    def make_box(self, dims_mm, yaw_deg=0.0):
        L, W, H = dims_mm

        points = self._make_box_points(
            dims_mm,
            self.cfg.object_points
        )

        # Rotate around Z.
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

    def make_sphere(self, dims_mm, yaw_deg=0.0):
        """
        dims_mm = (D, D, D)
        """
        D = dims_mm[0]
        radius = D / 2.0

        n = self.cfg.object_points

        theta = self.rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(
            1 - 2 * self.rng.random(n)
        )

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points = np.column_stack([x, y, z])

        R = self._rotation_z(yaw_deg)
        points = points @ R.T

        points[:, 2] -= points[:, 2].min()

        points += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=points.shape,
        )

        return points

    def make_bottle(self, dims_mm, yaw_deg=0.0):
        """
        Box + little cylinder on top
        """
        L, W, H = dims_mm

        box_h = H * 0.75
        box = self.make_box((L, W, box_h), yaw_deg)

        radius = min(L, W) * 0.12
        height = H - box_h

        n_total = self.cfg.object_points // 20

        side_area = 2.0 * np.pi * radius * height
        top_area = np.pi * radius ** 2
        n_side = int(round(n_total * side_area / (side_area + top_area)))
        n_top = n_total - n_side

        theta_side = self.rng.uniform(0, 2 * np.pi, n_side)
        side = np.column_stack([
            radius * np.cos(theta_side),
            radius * np.sin(theta_side),
            box_h + self.rng.random(n_side) * height,
        ])

        theta_top = self.rng.uniform(0, 2 * np.pi, n_top)
        r_top = radius * np.sqrt(self.rng.random(n_top))
        top = np.column_stack([
            r_top * np.cos(theta_top),
            r_top * np.sin(theta_top),
            np.full(n_top, H),
        ])

        bump = np.vstack([side, top])

        R = self._rotation_z(yaw_deg)
        bump = bump @ R.T
        bump += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=bump.shape,
        )

        return np.vstack([box, bump])

    def make_cylinder(self, dims_mm, yaw_deg=0.0):
        """
        dims_mm = (D, D, H)
        """
        L, _, H = dims_mm
        radius = L / 2.0

        n_total = self.cfg.object_points


        side_area = 2.0 * np.pi * radius * H
        top_area = np.pi * radius ** 2
        n_side = int(round(n_total * side_area / (side_area + top_area)))
        n_top = n_total - n_side

        theta_side = self.rng.uniform(0, 2 * np.pi, n_side)
        side = np.column_stack([
            radius * np.cos(theta_side),
            radius * np.sin(theta_side),
            self.rng.random(n_side) * H,
        ])

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

    def make_pencil(self, dims_mm, yaw_deg=0.0):
        """
        Cylinder with a cone on top.
        dims_mm = (D, D, H)
        """
        D = dims_mm[0]
        H = dims_mm[2]

        radius = D / 2.0

        cylinder_h = H * 0.7
        cone_h = H * 0.3

        n_total = self.cfg.object_points

        cylinder_area = 2.0 * np.pi * radius * cylinder_h
        top_area = np.pi * radius ** 2
        cone_slant = np.sqrt(radius ** 2 + cone_h ** 2)
        cone_area = np.pi * radius * cone_slant

        total_area = cylinder_area + top_area + cone_area

        n_cylinder = int(round(n_total * cylinder_area / total_area))
        n_top = int(round(n_total * top_area / total_area))
        n_cone = n_total - n_cylinder - n_top

        # Боковая поверхность цилиндра
        theta = self.rng.uniform(0, 2 * np.pi, n_cylinder)
        cylinder = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta),
            self.rng.random(n_cylinder) * cylinder_h,
        ])

        # Верхняя круглая грань цилиндра
        theta = self.rng.uniform(0, 2 * np.pi, n_top)
        r = radius * np.sqrt(self.rng.random(n_top))

        top = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.full(n_top, cylinder_h),
        ])

        # Боковая поверхность конуса
        theta = self.rng.uniform(0, 2 * np.pi, n_cone)
        z = cylinder_h + self.rng.random(n_cone) * cone_h

        # Радиус сечения конуса уменьшается к вершине
        local_r = radius * (1 - (z - cylinder_h) / cone_h)

        cone = np.column_stack([
            local_r * np.cos(theta),
            local_r * np.sin(theta),
            z,
        ])

        points = np.vstack([cylinder, top, cone])

        # Поворот вокруг Z
        R = self._rotation_z(yaw_deg)
        points = points @ R.T

        # Габарит по высоте уже начинается с z = 0
        points[:, 2] -= points[:, 2].min()

        # Шум глубины
        points += self.rng.normal(
            scale=self.cfg.depth_noise_std_mm,
            size=points.shape,
        )

        return points   