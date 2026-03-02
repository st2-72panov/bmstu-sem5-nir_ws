import cv2
import cv2.aruco as aruco
import os

# Параметры
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
start_id = 101       # Начальный ID
marker_count = 9     # Количество маркеров
size = 500           # Размер в пикселях

# Создаём папку для маркеров
# output_dir = "markers"
output_dir = "markers"
os.makedirs(output_dir, exist_ok=True)

for i in range(marker_count):
    marker_id = start_id + i
    
    # Генерация (совместимо со старыми и новыми версиями OpenCV)
    try:
        img = aruco.generateImageMarker(aruco_dict, marker_id, size)
    except AttributeError:
        img = aruco.drawMarker(aruco_dict, marker_id, size)
    
    # Сохранение
    filename = f"{output_dir}/marker_{marker_id}.png"
    cv2.imwrite(filename, img)
    # print(f"✓ Маркер {marker_id} сохранен как {filename}")

print(f"\nГотово! Сгенерировано {marker_count} маркеров (ID {start_id}-{start_id + marker_count - 1})")