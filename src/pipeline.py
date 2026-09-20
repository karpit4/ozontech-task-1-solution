from dataclasses import dataclass
import open3d as o3d
import numpy as np

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
        o3d.utility.random.seed(cfg.seed)

    def remove_conveyor_plane(self, cloud):
        distance = self.cfg.ransac_distance_mm / 1000.0

        plane_model, _ = cloud.segment_plane(
            distance_threshold=distance,
            ransac_n=3,
            num_iterations=self.cfg.ransac_iterations,
        )

        # Нормируем (a, b, c, d) так, чтобы |n| = 1
        plane_model = np.asarray(plane_model, dtype=float)
        plane_model /= np.linalg.norm(plane_model[:3])

        # Камера смотрит сверху, поэтому нормаль должна смотреть вверх (+Z)
        if plane_model[2] < 0:
            plane_model = -plane_model

        normal = plane_model[:3]
        points = np.asarray(cloud.points)

        # Расстояние со знаком от каждой точки до плоскости ленты
        signed_distance = points @ normal + plane_model[3]

        # Оставляем только точки выше ленты (допуск на шум глубины)
        keep = signed_distance > distance

        object_cloud = cloud.select_by_index(np.where(keep)[0])
        removed = np.where(~keep)[0]

        return object_cloud, plane_model, removed

    def filter_outliers(self, cloud):
        if len(cloud.points) == 0:
            return cloud

        # Downsample before expensive operations.
        voxel = self.cfg.voxel_size_mm / 1000.0
        filtered = cloud.voxel_down_sample(voxel)

        # 1. Радиусный фильтр с адаптивным радиусом. Масштаб берём из самого
        #    облака (медиана расстояний до ближайшего соседа): медиана не
        #    "слепнет" от выбросов, в отличие от среднего и σ в SOR.
        min_neighbors = self.cfg.radius_outlier_min_neighbors
        if len(filtered.points) > min_neighbors + 1:
            nn = np.asarray(filtered.compute_nearest_neighbor_distance())
            radius = self.cfg.radius_outlier_scale * float(np.median(nn))
            if radius > 0.0:
                filtered, _ = filtered.remove_radius_outlier(
                    nb_points=min_neighbors,
                    radius=radius,
                )

        # 2. SOR: мелкая чистка того, что осталось.
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

        labels = np.asarray(
            cloud.cluster_dbscan(
                eps = self.cfg.cluster_eps_mm / 1000,
                min_points = self.cfg.cluster_min_points,
                print_progress=False,
            )
        )

        # -1 означает шум, его не учитываем
        valid_labels = labels[labels >= 0]

        if len(valid_labels) == 0:
            return cloud

        # Считаем количество точек в каждом кластере
        unique_labels, counts = np.unique(
            valid_labels,
            return_counts=True,
        )

        # Находим крупнейший кластер
        largest_label = unique_labels[np.argmax(counts)]

        indices = np.where(labels == largest_label)[0]
        newcloud = cloud.select_by_index(indices)
        return newcloud

    def minimum_obb(self, cloud, plane_normal):
        """
        Open3D MINIMAL_JYLANKI.

        plane_normal - единичная нормаль ленты, направленная вверх (+Z).
        По ней находим вертикальную ось бокса, чтобы поправить только высоту.
        """
        if len(cloud.points) < 4:
            raise ValueError("Not enough points for OBB.")

        hull, _ = cloud.compute_convex_hull()
        hull.compute_vertex_normals()

        obb = hull.get_minimal_oriented_bounding_box(
            robust=True
        )

        # Open3D: center и extent в метрах, столбцы R - оси бокса.
        center = np.asarray(obb.center, dtype=float)
        R = np.asarray(obb.R, dtype=float)
        extent = np.asarray(obb.extent, dtype=float).copy()

        # Поправка на срез основания: слой толщиной ~ransac_distance_mm мы
        # выбросили вместе с лентой, поэтому высота систематически занижена.
        correction = self.cfg.height_correction_mm / 1000.0
        if correction != 0.0:
            # Вертикальная ось бокса - та, что лучше всего совпадает с
            # нормалью ленты (а не просто наименьший размер).
            alignment = R.T @ np.asarray(plane_normal, dtype=float)
            k = int(np.argmax(np.abs(alignment)))
            up = R[:, k] * np.sign(alignment[k])  # эта ось, направленная вверх

            # Растим бокс вниз, к ленте: там и лежал срезанный слой.
            extent[k] += correction
            center = center - 0.5 * correction * up

            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

        # Return sorted dimensions. The semantic labels L/W/H are assigned
        # by descending size for this demo.
        dims_mm = np.sort(extent * 1000.0)[::-1]

        return Measurement(
            length_mm=float(dims_mm[0]),
            width_mm=float(dims_mm[1]),
            height_mm=float(dims_mm[2]),
        ), obb, hull

    def measure(self, raw_cloud):
        object_cloud, plane_model, plane_inliers = \
            self.remove_conveyor_plane(raw_cloud)

        filtered = self.filter_outliers(object_cloud)
        # filtered = self.keep_main_component(filtered)
        measurement, obb, hull = self.minimum_obb(filtered, plane_model[:3])

        return {
            "measurement": measurement,
            "cloud_after_plane": object_cloud,
            "cloud_filtered": filtered,
            "plane_model": plane_model,
            "plane_inliers": plane_inliers,
            "obb": obb,
            "hull": hull,
        }
