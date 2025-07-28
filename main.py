import cv2
import numpy as np

from detector_screw import DetectorScrew
from detector_circle import DetectorCircle

TARGET_DISTANCE = 130  # 目标距离圆心的距离
TARGET_Z = 0  # 目标距离平面的高度
SCREW_DISTANCE = 40  # 螺丝距离圆心的距离
ANGLE_RAD = np.radians(60)

# 目标点和螺丝点的世界坐标
TARGET_POINTS_WORLD = np.array(
    [
        [0, TARGET_DISTANCE, TARGET_Z],
        [TARGET_DISTANCE * np.sin(ANGLE_RAD), TARGET_DISTANCE * np.cos(ANGLE_RAD), TARGET_Z],
        [TARGET_DISTANCE * np.sin(ANGLE_RAD), -TARGET_DISTANCE * np.cos(ANGLE_RAD), TARGET_Z], 
        [0, -TARGET_DISTANCE, TARGET_Z],
        [-TARGET_DISTANCE * np.sin(ANGLE_RAD), -TARGET_DISTANCE * np.cos(ANGLE_RAD), TARGET_Z],
        [-TARGET_DISTANCE * np.sin(ANGLE_RAD), TARGET_DISTANCE * np.cos(ANGLE_RAD), TARGET_Z],
    ],
    dtype=np.float32,
)
SCREW_POINTS_WORLD = np.array(
    [
        [0.0, -SCREW_DISTANCE, 0.0],  # 上
        [0.0, SCREW_DISTANCE, 0.0],  # 下
        [-SCREW_DISTANCE, 0.0, 0.0],  # 左
        [SCREW_DISTANCE, 0.0, 0.0],  # 右
    ],
    dtype=np.float32,
)

# 相机参数
CAMERA_MATRIX = np.array(
    [[690.689209, 0, 643.113159], [0, 690.479431, 360.236084], [0, 0, 1]],
    dtype=np.float32,
)
DIST_COEFFS = np.array(
    [
        0.008123,
        -0.048731,
        0.000216,
        0.000457,
        0.033166,
        0.000000,
        0.000000,
        0.000000,
    ],
    dtype=np.float32,
)


def screw_test(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"can not read: {image_path}")
        return

    detector = DetectorScrew(TARGET_POINTS_WORLD, SCREW_POINTS_WORLD, CAMERA_MATRIX, DIST_COEFFS)
    print(detector.process_frame(image))

def circle_test(image_rgb_path: str, image_depth_path: str):
    image_rgb = cv2.imread(image_rgb_path)
    image_depth = cv2.imread(image_depth_path, cv2.IMREAD_UNCHANGED)
    if image_rgb is None or image_depth is None:
        print(f"can not read: {image_rgb_path} or {image_depth_path}")
        return

    detector = DetectorCircle(TARGET_Z, CAMERA_MATRIX, DIST_COEFFS)
    print(detector.process_frame(image_rgb, image_depth))

if __name__ == "__main__":
    # screw_test("data/frame2/color_000000_rgb.png")
    circle_test("data/frame2/color_000000_rgb.png", "data/frame2/raw_depth_000000.png")
