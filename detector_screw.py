import cv2
import numpy as np
from typing import List, Tuple, Optional


class DetectorScrew:
    """单位mm"""

    def __init__(self, target_points_world, screw_points_world, camera_matrix, dist_coeffs, debug=True):
        self.target_points_world = target_points_world
        self.screw_points_world = screw_points_world

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.debug = debug

    def detect_screws(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        检测图像中的四个螺丝，返回螺丝中心坐标

        Args:
            image: 输入图像

        Returns:
            四个螺丝中心的像素坐标按上、下、左、右顺序
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.GaussianBlur(gray, (5, 5), 2)
        _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        if self.debug:
            cv2.imshow("gray", gray)
            cv2.waitKey(0)

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screw_circles = []
        black_bottle = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            # 周长
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # 圆度
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.8:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 5 <= radius <= 20:
                    screw_circles.append((int(x), int(y), int(radius)))
                elif radius > 100:
                    black_bottle = (int(x), int(y), int(radius))

        if self.debug:
            for x, y, r in screw_circles:
                cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            if black_bottle:
                x, y, r = black_bottle
                cv2.circle(image, (x, y), r, (255, 0, 0), 2)
            cv2.imshow("detected_circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if len(screw_circles) != 4:
            print(f"screws number invalid: {len(screw_circles)}")
            return None
        if not black_bottle:
            print("no black bottle detected")
            return None

        dist2bottle = [np.sqrt((x - black_bottle[0]) ** 2 + (y - black_bottle[1]) ** 2) for (x, y, _) in screw_circles]

        # sort by distance to black bottle
        sorted_indices = np.argsort(dist2bottle)
        left = screw_circles[sorted_indices[0]][:2]
        bottom = screw_circles[sorted_indices[1]][:2]
        top = screw_circles[sorted_indices[2]][:2]
        right = screw_circles[sorted_indices[3]][:2]
        if not all([top, bottom, left, right]):
            print("screw detection failed")
            return None

        if self.debug:
            screw_labels = ["T", "B", "L", "R"]
            for i, (x, y) in enumerate([top, bottom, left, right]):
                cv2.putText(image, screw_labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("image", image)

        return [top, bottom, left, right]

    def estimate_pose_pnp(
        self,
        image_points: List[Tuple[int, int]],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用PnP算法估计位姿

        Args:
            image_points: 图像中的点坐标
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数

        Returns:
            (rvec, tvec) 旋转向量和平移向量
        """

        image_points_np = np.array([[float(x), float(y)] for x, y in image_points], dtype=np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.screw_points_world,
                image_points_np,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success:
                return rvec, tvec
            else:
                print("pnp failed")
                return None

        except Exception as e:
            print(f"pnp calculation error: {e}")
            return None

    def process_frame(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        处理单帧图像的完整流程

        Args:
            image: 输入图像

        Returns:
            y_max_target: 下方的目标点的像素坐标
        """
        screws_image_points = self.detect_screws(image)
        if screws_image_points is None:
            return None

        pnp_result = self.estimate_pose_pnp(screws_image_points, self.camera_matrix, self.dist_coeffs)
        if pnp_result is None:
            return None
        rvec, tvec = pnp_result
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        target_image_points = []  # 像素坐标系下的
        target_camera_points = []  # 相机坐标系下的
        for target_world_point in self.target_points_world:
            target_camera_point = rotation_matrix @ target_world_point + tvec.reshape(3)
            target_camera_points.append(target_camera_point)

            target_pose_image_points, _ = cv2.projectPoints(
                target_world_point,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
            target_image_points.append(tuple(map(int, target_pose_image_points[0][0])))

        y_max_target = max(target_image_points, key=lambda p: p[1])
        if self.debug:
            for point in target_image_points:
                cv2.circle(image, point, 10, (0, 255, 0), -1)
            cv2.circle(image, y_max_target, 10, (255, 255, 0), -1)
            cv2.imshow("target_pose", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return y_max_target
