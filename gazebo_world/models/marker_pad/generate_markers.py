import cv2
import cv2.aruco as aruco
import os
import numpy as np

# ================= КОНФИГУРАЦИЯ =================
MARKER_COUNT = 9
START_ID = 101
MARKER_DICT = aruco.DICT_6X6_250

PIXEL_WIDTH = 100
BORDER_WIDTH = PIXEL_WIDTH * 1
GRID_WIDTH = (6 + 2) * PIXEL_WIDTH  # Размер самого маркера с внутренними отступами

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "materials/textures/markers")

# ================= ГЕНЕРАЦИЯ МАРКЕРОВ =================
os.makedirs(OUTPUT_DIR, exist_ok=True)
aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
marker_images = []

for marker_id in range(START_ID, START_ID + MARKER_COUNT):
    try:
        img = aruco.generateImageMarker(aruco_dict, marker_id, GRID_WIDTH)
    except Exception:
        img = aruco.drawMarker(aruco_dict, marker_id, GRID_WIDTH)
    
    # Добавляем белую каёмочку
    img_size = img.shape[0]  # Размер текущего изображения
    bordered_size = img_size + 2 * BORDER_WIDTH  # Размер с каёмочкой
    bordered_img = 255 * np.ones((bordered_size, bordered_size), dtype=np.uint8)  # Белый фон
    bordered_img[BORDER_WIDTH:BORDER_WIDTH + img_size, 
                 BORDER_WIDTH:BORDER_WIDTH + img_size] = img  # Вставка маркера
    
    filename = f"{OUTPUT_DIR}/{marker_id}.png"
    cv2.imwrite(filename, bordered_img)
    marker_images.append((marker_id, bordered_img))

print(f"✓ Маркеры сгенерированы")
