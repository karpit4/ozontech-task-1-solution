from dataclasses import dataclass
from typing import Tuple 

@dataclass
class Config:
    #Random-state
    seed : int = 6234237
    #Object parameters
    object_shape : str = "bottle"
    object_dims : Tuple[float,float,float] = (
        150,        
        150,
        700
    ) #L, W, H
    object_yaw : float  = 0
    
    # Conveyour parameters
    conveyor_width_mm: float = max(object_dims[0],object_dims[1]) + 200
    conveyor_length_mm: float = conveyor_width_mm

    # Camera parameters
    camera_height_mm: float = 600.0
    depth_noise_std_mm: float = 0.5

    # Point cloud
    object_points: int = 5000
    conveyor_points: int = 3000
    outlier_points: int = 40

    # Filtering
    voxel_size_mm: float = 2
    ransac_distance_mm: float = 2
    ransac_iterations: int = 1000
    sor_neighbors: int = 30
    sor_std_ratio: float = 1.5
    
    height_correction_mm: float = 3.0
    
    # Outlier removal: radius filter (runs before SOR)
    radius_outlier_scale: float = 3.0       # радиус = scale × медиана расстояний до ближайшего соседа
    radius_outlier_min_neighbors: int = 6   # минимум соседей в этом радиусе

    #Clustering (keep_main_component)
    cluster_eps_mm : float = 32
    cluster_min_points : int = 20

    # Quality
    min_object_points: int = 100

    # Visualization
    save_visualization: bool = True
