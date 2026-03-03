import cv2
import cv2.aruco as aruco
import os

# ================= КОНФИГУРАЦИЯ =================
MARKER_COUNT = 9
START_ID = 101
MARKER_DICT = aruco.DICT_6X6_250
GRID_SIZE = (6 + 2) * 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "materials/textures/markers")

# ================= ГЕНЕРАЦИЯ МАРКЕРОВ =================
os.makedirs(OUTPUT_DIR, exist_ok=True)
aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)

marker_images = []
for marker_id in range(START_ID, START_ID + MARKER_COUNT):
    try:
        img = aruco.generateImageMarker(aruco_dict, marker_id, GRID_SIZE)
    except Exception:
        img = aruco.drawMarker(aruco_dict, marker_id, GRID_SIZE)
    
    filename = f"{OUTPUT_DIR}/{marker_id}.png"
    cv2.imwrite(filename, img)
    marker_images.append((marker_id, img))
    
print(f"✓ Маркеры сгенерированы")
