import cv2
import numpy as np
from typing import Tuple, Optional


class DetectorCircle:
    def __init__(self, target_z, camera_matrix, dist_coeffs, debug=True):
        self.target_z = target_z
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.debug = debug

    def detect_circle(self, image: np.ndarray) -> Tuple[int, int, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 2)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        gray = cv2.bitwise_not(gray)
        gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        if self.debug:
            cv2.imshow("gray", gray)
            cv2.waitKey(0)

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 200 or perimeter < 100:
                continue

            roundness = (4 * np.pi * area) / (perimeter * perimeter)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if roundness < 0.8 or radius > 100:
                continue
        
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            candidates.append((int(x), int(y), int(radius)))

        assert len(candidates) > 0, "No valid circles found"

        if self.debug:
            for candidate in candidates:
                x, y, radius = candidate
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)

        return max(candidates, key=lambda c: c[1])

    def get_depth(self, image_depth: np.ndarray, circle: Tuple[int, int, int]) -> float:
        x, y, radius = circle
        h, w = image_depth.shape
        inner_r = int(radius * 0.95)

        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask = (dist_from_center <= radius) & (dist_from_center >= inner_r)
        ring_depths = image_depth[mask]

        valid_depths = ring_depths[(ring_depths > 0) & (~np.isnan(ring_depths))]
        if valid_depths.size == 0:
            return float("nan")

        return float(np.mean(valid_depths))

    def convert_to_3d(self, x: int, y: int, depth: float) -> np.ndarray:
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        if np.isnan(depth):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth + self.target_z

        return np.array([X, Y, Z], dtype=np.float32)

    def process_frame(self, image_rgb: np.ndarray, image_depth: np.ndarray) -> Optional[np.ndarray]:
        print("Processing frame: image_rgb shape:", image_rgb.shape, "image_depth shape:", image_depth.shape)
        result = self.detect_circle(image_rgb)

        print(f"circle center: {result[0]}, {result[1]}, radius: {result[2]}")

        if self.debug:
            x, y, radius = result
            cv2.circle(image_rgb, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(image_rgb, (x, y), int(radius * 0.95), (0, 255, 0), 2)
            cv2.circle(image_rgb, (x, y), 5, (255, 255, 0), -1)
            cv2.imshow("Detected Circle", image_rgb)
            cv2.waitKey(0)
        
        depth = self.get_depth(image_depth, result)
        print("circle depth:", depth)


        return self.convert_to_3d(x, y, depth)

