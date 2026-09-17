from dataclasses import dataclass


@dataclass
class Config:
    # Размеры конвейера
    conveyor_width_mm: float = 600.0
    conveyor_length_mm: float = 1200.0

    # Имитируемая камера
    camera_height_mm: float = 600.0
    depth_noise_std_mm: float = 1.5

    # Point cloud
    object_points: int = 1200
    conveyor_points: int = 900
    outlier_points: int = 15

    # Filtering
    voxel_size_mm: float = 2.0
    ransac_distance_mm: float = 4.0
    ransac_iterations: int = 1000
    sor_neighbors: int = 30
    sor_std_ratio: float = 2.0

    # Quality
    min_object_points: int = 100

    # Visualization
    save_visualization: bool = True
