import os
from typing import List, Tuple

import cv2
import numpy as np

# Константы из stag_detector.cpp
HALF_PI = 1.570796326794897
INNER_CIRCLE_RADIUS = 0.4 * 0.9  # outerCircleRadius * 0.9

# Параметры для 12 позиций в каждом квадранте (radius, angle)
CIRCLE_PARAMS = [
    (0.088363142525988, 0.785398163397448),
    (0.206935928182607, 0.459275804122858),
    (0.206935928182607, HALF_PI - 0.459275804122858),
    (0.313672146827381, 0.200579720495241),
    (0.327493143484516, 0.591687617505840),
    (0.327493143484516, HALF_PI - 0.591687617505840),
    (0.313672146827381, HALF_PI - 0.200579720495241),
    (0.437421957035861, 0.145724938287167),
    (0.437226762361658, 0.433363129825345),
    (0.430628029742607, 0.785398163397448),
    (0.437226762361658, HALF_PI - 0.433363129825345),
    (0.437421957035861, HALF_PI - 0.145724938287167),
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(SCRIPT_DIR, "stag.png")

class STag:
    def __init__(self):
        self.circle_centers = self._compute_circle_centers()
        pattern_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
        pattern_img = self._extract_marker_region(pattern_img)
        self.pattern = self._decode_pattern(pattern_img)
        
    def is_valid(self, pattern_img: np.ndarray) -> bool:
        """
        Проверяет, совпадает ли паттерн на изображении с эталонным.
        Проверяет все 4 поворота против часовой стрелки.
        """
        if self.pattern is None:
            raise ValueError("Reference pattern is not set.")
        pattern_img = self._extract_marker_region(pattern_img)
        
        # Проверяем все 4 поворота
        for rotation in range(4):
            # Поворачиваем изображение на 90 * rotation градусов против ч.с.
            rotated_img = np.rot90(pattern_img, k=rotation)
            
            # Декодируем паттерн
            decoded_pattern = self._decode_pattern(rotated_img)
            
            # Сравниваем с эталонным паттерном
            if self._patterns_match(self.pattern, decoded_pattern):
                return True
        
        return False
    
    def _extract_marker_region(self, img: np.ndarray) -> np.ndarray:
        # Обрезка чёрного фона. Остаётся только круговая часть маркера
        h, w = img.shape[:2]
        l, r = int(0.1 * w), int(0.9 * w)
        return img[l:r, l:r]
    
    def _compute_circle_centers(self) -> List[Tuple[float, float]]:
        """
        Вычисляет координаты 48 кружков в нормализованной системе (0-1).
        Порядок: для каждого квадранта (0-3), для каждой позиции (0-11).
        Как в fillCodeLocations() из stag_detector.cpp
        """
        centers = []
        
        for quadrant in range(4):
            for pos in range(12):
                radius, base_angle = CIRCLE_PARAMS[pos]
                angle = base_angle + quadrant * HALF_PI
                
                x = 0.5 + np.cos(angle) * radius * (INNER_CIRCLE_RADIUS / 0.5)
                y = 0.5 - np.sin(angle) * radius * (INNER_CIRCLE_RADIUS / 0.5)
                
                centers.append((x, y))
        
        return centers
    
    def _decode_pattern(self, pattern_img: np.ndarray) -> List[List[int]]:
        """
        Вычисляет бинарный паттерн из 48 кружков.
        Возвращает список из 4 списков по уровням.
        """
        if pattern_img is None:
            raise ValueError("Image is None")
        height, width = pattern_img.shape
        
        # Конвертируем в BGR для отладочного рисунка
        debug_img = cv2.cvtColor(pattern_img, cv2.COLOR_GRAY2BGR)
        
        # Получаем средние значения для всех 48 кружков
        # Порядок: квадрант 0 (позиции 0-11), квадрант 1 (позиции 0-11), ...
        circle_values = []
        
        for idx, (norm_x, norm_y) in enumerate(self.circle_centers):
            # Преобразуем нормализованные координаты в пиксели
            px = int(norm_x * width)
            py = int(norm_y * height)
            
            # Вычисляем среднее значение в точке
            radius_pixels = 12
            y_min = max(0, py - radius_pixels)
            y_max = min(height, py + radius_pixels)
            x_min = max(0, px - radius_pixels)
            x_max = min(width, px + radius_pixels)
            
            region = pattern_img[y_min:y_max, x_min:x_max]
            avg_value = np.mean(region)
            
            # Как в stag_detector.cpp: THRESH_BINARY_INV
            # Чёрные кружки (< 128) = 0, белые (>= 128) = 1
            value = 0 if avg_value < 128 else 1
            circle_values.append(value)
            
            # Рисуем точку для отладки
            # Красная = белый кружок (value=0), Синяя = чёрный кружок (value=1)
            color = (0, 0, 255) if value == 1 else (255, 0, 0)  # BGR формат
            cv2.circle(debug_img, (px, py), 5, color, -1)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "img_debug.png"), debug_img)
        
        # Группируем по уровням:
        # Индексы в circle_values:
        # Квадрант 0: 0-11, Квадрант 1: 12-23, Квадрант 2: 24-35, Квадрант 3: 36-47
        pattern = []
        
        # Уровень 0: позиция 0 каждого квадранта (индексы 0, 12, 24, 36)
        level0 = [circle_values[0 + i * 12] for i in range(4)]
        pattern.append(level0)
        
        # Уровень 1: позиции 1-2 каждого квадранта (индексы 1,2, 13,14, 25,26, 37,38)
        level1 = []
        for i in range(4):
            level1.append(circle_values[1 + i * 12])
            level1.append(circle_values[2 + i * 12])
        pattern.append(level1)
        
        # Уровень 2: позиции 3-6 каждого квадранта (индексы 3-6, 15-18, 27-30, 39-42)
        level2 = []
        for i in range(4):
            for pos in range(3, 7):
                level2.append(circle_values[pos + i * 12])
        pattern.append(level2)
        
        # Уровень 3: позиции 7-11 каждого квадранта (индексы 7-11, 19-23, 31-35, 43-47)
        level3 = []
        for i in range(4):
            for pos in range(7, 12):
                level3.append(circle_values[pos + i * 12])
        pattern.append(level3)
        
        return pattern
    
    def _patterns_match(self, pattern1: List[List[int]], pattern2: List[List[int]]) -> bool:
        """Сравнивает два паттерна"""
        if len(pattern1) != len(pattern2):
            return False
        
        for level1, level2 in zip(pattern1, pattern2):
            if len(level1) != len(level2):
                return False
            for val1, val2 in zip(level1, level2):
                if val1 != val2:
                    return False
        return True


if __name__ == "__main__":
    stag = STag()
    
    # Проверяем другой паттерн
    test_img = cv2.imread(os.path.join(SCRIPT_DIR, "stag.png"), cv2.IMREAD_GRAYSCALE)
    is_valid = stag.is_valid(test_img)
    
    print(f"Pattern is valid: {is_valid}")
    
    # Можно также получить декодированный паттерн
    pattern = stag._decode_pattern(test_img)
    print("Decoded pattern:")
    for i, level in enumerate(pattern):
        print(f"Level {i}: {level}")
