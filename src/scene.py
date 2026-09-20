import math
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .objects import ObjectGenerator

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
    Synthetic scene:
      z = 0                 -> conveyor
      object sits on z=0
      object is rotated around Z
    """
    SHAPES = ("box", "cylinder", "bottle","sphere","pencil")
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.objects = ObjectGenerator(cfg)
        self.rng = np.random.default_rng(cfg.seed)


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
        
        if shape in ("cylinder","pencil"):
            dims_mm = (dims_mm[0],dims_mm[0],dims_mm[2])
        elif shape == "sphere":
            dims_mm = (dims_mm[0],dims_mm[0],dims_mm[0])
            
        obj = self.objects.create(shape, dims_mm, yaw_deg)
        
        gt = GroundTruth(
            length_mm=dims_mm[0],
            width_mm=dims_mm[1],
            height_mm=dims_mm[2],
            yaw_deg=yaw_deg,
        )


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
