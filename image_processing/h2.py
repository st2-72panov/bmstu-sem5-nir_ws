import cv2
import numpy as np

# === ТО ЖЕ САМОЕ исходное изображение ===
img = np.zeros((500, 600, 3), dtype=np.uint8) + 255
pts = np.array([[150, 100], [450, 120], [420, 400], [180, 380]], np.int32)

for y in range(500):
    for x in range(600):
        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
            img[y, x] = [x % 256, (x + y) % 256, y % 256]

cv2.polylines(img, [pts], True, (0, 0, 255), 2)

# === Гомография: четырёхугольник → квадрат ===
dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
M = cv2.getPerspectiveTransform(np.float32(pts), dst_pts)

# === Находим, куда попадают углы ВСЕГО изображения ===
h, w = img.shape[:2]
img_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
img_corners_trans = cv2.perspectiveTransform(img_corners.reshape(-1, 1, 2), M).reshape(-1, 2)

# === Объединяем с углами квадрата для расчёта bounding box ===
all_points = np.vstack([dst_pts, img_corners_trans])
x_min, y_min = all_points.min(axis=0)
x_max, y_max = all_points.max(axis=0)

# === Сдвиг, чтобы всё было в положительной области ===
translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
M_shifted = translate @ M

# === Размер выхода — весь bounding box ===
out_w, out_h = int(x_max - x_min), int(y_max - y_min)

# === Трансформируем ВСЁ изображение ===
warped_full = cv2.warpPerspective(img, M_shifted, (out_w, out_h))

# === Показываем ===
cv2.imshow('Original (whole scene)', img)
cv2.imshow('Warped (whole scene, nothing cropped)', warped_full)
cv2.waitKey(0)
cv2.destroyAllWindows()