from dataclasses import dataclass


@dataclass
class Config:
    #Object parameters
    OBJECT_DIMS = (300, 300, 300) #L,W,H
    OBJECT_YAW = 30
    
    # Conveyour parameters
    conveyor_width_mm: float = 600.0
    conveyor_length_mm: float = 1200.0

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

    # Quality
    min_object_points: int = 100

    # Visualization
    save_visualization: bool = True
