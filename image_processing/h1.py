import cv2
import numpy as np

# Создаём изображение с градиентом
img = np.zeros((500, 600, 3), dtype=np.uint8) + 255
pts = np.array([[150, 100], [450, 120], [420, 400], [180, 380]], np.int32)

# Рисуем градиентный четырёхугольник
for y in range(500):
    for x in range(600):
        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
            img[y, x] = [x % 256, (x + y) % 256, y % 256]

cv2.polylines(img, [pts], True, (0, 0, 255), 2)
for i, pt in enumerate(pts):
    cv2.circle(img, tuple(pt), 5, (255, 0, 0), -1)

# Гомография: четырёхугольник → квадрат
dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
M = cv2.getPerspectiveTransform(np.float32(pts), dst_pts)
warped = cv2.warpPerspective(img, M, (400, 400))

# Показываем результаты
cv2.imshow('Original', img)
cv2.imshow('Warped', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()