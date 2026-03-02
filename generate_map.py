#!/usr/bin/env python3
"""
Скрипт добавляет 9 ArUco маркеров (6x6) в мир Gazebo SDF
Каждый маркер состоит из физических блоков (чёрные/белые)
"""

import cv2
import cv2.aruco as aruco
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ================= КОНФИГУРАЦИЯ =================
MARKER_COUNT = 9
START_ID = 101
MARKER_DICT = aruco.DICT_6X6_250
GRID_SIZE = 6  # 6x6 для этого словаря

# Размеры в метрах (Gazebo)
BLOCK_SIZE = 0.1      # Размер одного блока маркера (10 см)
BLOCK_HEIGHT = 0.01   # Высота блоков (1 см - низко)
MARKER_GAP = 2        # Расстояние между маркерами в блоках

# Позиция первого маркера в мире
START_X = -3.0
START_Y = -3.0
START_Z = 0.015       # Чуть выше земли

# Файлы
WORLD_FILE = "uav_world.sdf"
MARKERS_DIR = "markers"
OUTPUT_FILE = "world_with_markers.sdf"

# ================= ГЕНЕРАЦИЯ МАРКЕРОВ =================
def generate_marker_images():
    """Генерирует изображения маркеров для чтения паттерна"""
    os.makedirs(MARKERS_DIR, exist_ok=True)
    aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
    
    marker_images = []
    for i in range(MARKER_COUNT):
        marker_id = START_ID + i
        try:
            img = aruco.generateImageMarker(aruco_dict, marker_id, GRID_SIZE * 100)
        except AttributeError:
            img = aruco.drawMarker(aruco_dict, marker_id, GRID_SIZE * 100)
        
        filename = f"{MARKERS_DIR}/marker_{marker_id}.png"
        cv2.imwrite(filename, img)
        marker_images.append((marker_id, img))
        print(f"✓ Маркер {marker_id} сгенерирован")
    
    return marker_images

# ================= СОЗДАНИЕ SDF БЛОКОВ =================
def get_block_color(pixel_value):
    """Определяет цвет блока по пикселю (чёрный или белый)"""
    # Если темно - чёрный блок, иначе белый
    return "black" if pixel_value < 128 else "white"

def create_marker_sdf(marker_id, grid_data, marker_index):
    """Создаёт SDF модель для одного маркера"""
    # Вычисляем позицию маркера в мире (расставляем в ряд)
    markers_per_row = 3
    row = marker_index // markers_per_row
    col = marker_index % markers_per_row
    
    # Смещение между маркерами
    marker_width = GRID_SIZE * BLOCK_SIZE
    gap_meters = MARKER_GAP * BLOCK_SIZE
    
    x = START_X + col * (marker_width + gap_meters)
    y = START_Y + row * (marker_width + gap_meters)
    z = START_Z
    
    # Создаём XML структуру для маркера
    model = ET.Element("model")
    model.set("name", f"aruco_marker_{marker_id}")
    
    pose = ET.SubElement(model, "pose")
    pose.text = f"{x} {y} {z} 0 0 0"
    
    static = ET.SubElement(model, "static")
    static.text = "true"
    
    link = ET.SubElement(model, "link")
    link.set("name", "link")
    
    # Создаём блоки для каждой ячейки 6x6
    for row_idx in range(GRID_SIZE):
        for col_idx in range(GRID_SIZE):
            # Получаем цвет ячейки из изображения
            pixel_y = int((row_idx + 0.5) * 100)
            pixel_x = int((col_idx + 0.5) * 100)
            pixel_val = grid_data[pixel_y, pixel_x]
            color = get_block_color(pixel_val)
            
            # Создаём визуальный блок
            visual = ET.SubElement(link, "visual")
            visual.set("name", f"block_{row_idx}_{col_idx}")
            
            visual_pose = ET.SubElement(visual, "pose")
            bx = (col_idx - GRID_SIZE/2 + 0.5) * BLOCK_SIZE
            by = (row_idx - GRID_SIZE/2 + 0.5) * BLOCK_SIZE
            visual_pose.text = f"{bx} {by} 0 0 0 0"
            
            geometry = ET.SubElement(visual, "geometry")
            box = ET.SubElement(geometry, "box")
            size = ET.SubElement(box, "size")
            size.text = f"{BLOCK_SIZE} {BLOCK_SIZE} {BLOCK_HEIGHT}"
            
            material = ET.SubElement(visual, "material")
            ambient = ET.SubElement(material, "ambient")
            diffuse = ET.SubElement(material, "diffuse")
            
            if color == "black":
                ambient.text = "0.05 0.05 0.05 1"
                diffuse.text = "0.05 0.05 0.05 1"
            else:
                ambient.text = "0.9 0.9 0.9 1"
                diffuse.text = "0.9 0.9 0.9 1"
    
    return model

# ================= РАБОТА С SDF ФАЙЛОМ =================
def add_markers_to_world(marker_models):
    """Добавляет маркеры в SDF файл мира"""
    # Читаем оригинальный файл
    tree = ET.parse(WORLD_FILE)
    root = tree.getroot()
    
    # Находим элемент world
    world = root.find("world")
    if world is None:
        world = root  # Если root уже world
    
    # Добавляем каждую модель маркера
    for model in marker_models:
        world.append(model)
    
    # Сохраняем с форматированием
    xml_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    # Убираем лишние пустые строки
    lines = [line for line in pretty_xml.split("\n") if line.strip()]
    clean_xml = "\n".join(lines)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(clean_xml)
    
    print(f"✓ Мир сохранён в {OUTPUT_FILE}")

# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def main():
    print("=" * 50)
    print("Генерация ArUco маркеров для Gazebo world.sdf")
    print("=" * 50)
    
    # Шаг 1: Генерируем изображения маркеров
    print("\n[1/3] Генерация изображений маркеров...")
    marker_images = generate_marker_images()
    
    # Шаг 2: Создаём SDF модели для каждого маркера
    print("\n[2/3] Создание SDF моделей...")
    marker_models = []
    for idx, (marker_id, img) in enumerate(marker_images):
        model = create_marker_sdf(marker_id, img, idx)
        marker_models.append(model)
        print(f"  ✓ Маркер {marker_id} готов")
    
    # Шаг 3: Добавляем в мир
    print("\n[3/3] Добавление маркеров в мир...")
    add_markers_to_world(marker_models)
    
    print("\n" + "=" * 50)
    print("ГОТОВО!")
    print(f"Создано {MARKER_COUNT} маркеров")
    print(f"Файл: {OUTPUT_FILE}")
    print(f"Запуск: gazebo {OUTPUT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()