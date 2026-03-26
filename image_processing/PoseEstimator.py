import cv2
import numpy as np

def estimate_pose(marker_corners):
    width = 1280
    height = 720
    h_fov = 1.047  # радианы

    cx = width / 2.0
    cy = height / 2.0
    fx = width / (2.0 * np.tan(h_fov / 2.0))
    fy = fx

    K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])
    D = np.array([0, 0, 0, 0, 0])
    marker_size = 0.5
    original_corners = np.array([
        [0, 0, 0],
        [marker_size, 0, 0],
        [marker_size, marker_size, 0],
        [0, marker_size, 0]
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        original_corners,
        marker_corners,
        K, 
        D, 
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        return None

    R_matrix, _ = cv2.Rodrigues(rvec)

    # 7. Формирование матрицы трансформации 4x4 (Pose Matrix)
    # Эта матрица описывает трансформацию: Из системы координат Маркера -> В систему координат Камеры
    T_marker_to_cam = np.eye(4)
    T_marker_to_cam[:3, :3] = R_matrix
    T_marker_to_cam[:3, 3] = tvec.flatten()

    # 8. (Опционально) Поза камеры относительно маркера
    T_cam_to_marker = np.linalg.inv(T_marker_to_cam)
    cam_position_in_marker_frame = T_cam_to_marker[:3, 3]

    return cam_position_in_marker_frame