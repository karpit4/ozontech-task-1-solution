from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class Measurement:
    length_mm: float
    width_mm: float
    height_mm: float

    @property
    def dimensions(self):
        return np.array(
            [self.length_mm, self.width_mm, self.height_mm],
            dtype=float,
        )


class DimensioningPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def remove_conveyor_plane(self, cloud):
        """
        RANSAC plane segmentation.
        Plane equation: ax + by + cz + d = 0.
        """
        distance = self.cfg.ransac_distance_mm / 1000.0

        plane_model, inliers = cloud.segment_plane(
            distance_threshold=distance,
            ransac_n=3,
            num_iterations=self.cfg.ransac_iterations,
        )

        object_cloud = cloud.select_by_index(inliers, invert=True)

        return object_cloud, plane_model, inliers

    def filter_outliers(self, cloud):
        if len(cloud.points) == 0:
            return cloud

        # Downsample before expensive operations.
        voxel = self.cfg.voxel_size_mm / 1000.0
        filtered = cloud.voxel_down_sample(voxel)

        if len(filtered.points) < self.cfg.sor_neighbors:
            return filtered

        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=self.cfg.sor_neighbors,
            std_ratio=self.cfg.sor_std_ratio,
        )

        return filtered

    def keep_main_component(self, cloud):
        """
        Удаляем мелкие компоненты после удаления конвейера.
        """
        if len(cloud.points) < self.cfg.min_object_points:
            return cloud

        labels, counts, _ = cloud.cluster_dbscan(
            eps=0.012,
            min_points=20,
            print_progress=False,
        )

        if len(counts) == 0:
            return cloud

        largest = int(np.argmax(counts))
        indices = np.where(labels == largest)[0]

        return cloud.select_by_index(indices)

    def minimum_obb(self, cloud):
        """
        Open3D MINIMAL_JYLANKI.
        """
        if len(cloud.points) < 4:
            raise ValueError("Not enough points for OBB.")

        hull, _ = cloud.compute_convex_hull()
        hull.compute_vertex_normals()

        obb = hull.get_minimal_oriented_bounding_box(
            robust=True
        )

        # Open3D extent is in meters.
        dims_mm = np.asarray(obb.extent) * 1000.0

        # Return sorted dimensions. The semantic labels L/W/H are assigned
        # by descending size for this demo.
        dims_mm = np.sort(dims_mm)[::-1]

        return Measurement(
            length_mm=float(dims_mm[0]),
            width_mm=float(dims_mm[1]),
            height_mm=float(dims_mm[2]),
        ), obb, hull

    def measure(self, raw_cloud):
        object_cloud, plane_model, plane_inliers = \
            self.remove_conveyor_plane(raw_cloud)

        filtered = self.filter_outliers(object_cloud)
        filtered = self.keep_main_component(filtered)

        measurement, obb, hull = self.minimum_obb(filtered)

        return {
            "measurement": measurement,
            "cloud_after_plane": object_cloud,
            "cloud_filtered": filtered,
            "plane_model": plane_model,
            "plane_inliers": plane_inliers,
            "obb": obb,
            "hull": hull,
        }
