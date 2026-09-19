from dataclasses import dataclass
from typing import Tuple 

@dataclass
class Config:
    #Random-state
    seed : int = 676767
    #Object parameters
    object_dims : Tuple[float,float,float] = (
        100,        
        100,
        100
    ) #L, W, H
    object_yaw : float  = 30
    object_shape : str = "bottle"
    
    # Conveyour parameters
    conveyor_width_mm: float = 600.0
    conveyor_length_mm: float = 600.0

    # Camera parameters
    camera_height_mm: float = 600.0
    depth_noise_std_mm: float = 1.5

    # Point cloud
    object_points: int = 5000
    conveyor_points: int = 3000
    outlier_points: int = 0

    # Filtering
    voxel_size_mm: float = 2.0
    ransac_distance_mm: float = 4.0
    ransac_iterations: int = 1000
    sor_neighbors: int = 30
    sor_std_ratio: float = 1.5

    #Clustering (keep_main_component)
    cluster_eps_mm : float = 32
    cluster_min_points : int = 20

    # Quality
    min_object_points: int = 100

    # Visualization
    save_visualization: bool = True
