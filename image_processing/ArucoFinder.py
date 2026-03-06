from datetime import datetime
import os
import time

import cv2
import numpy as np

from Aruco import Aruco
from MarkerFinder import MarkerFinder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_FOLDER = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")
FRAME_FACTOR = 2.0

class ArucoFinder(MarkerFinder):
    def __init__(self):
        super().__init__()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Step 1
    def _detect_and_filter_quads(self):
        """
        Шаг 1: Бинаризация, обнаружение четырёхугольников и фильтрация.
        """

        photo_gray = cv2.cvtColor(self.photo_cropped, cv2.COLOR_BGR2GRAY)
        _, photo_binary = cv2.threshold(photo_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(photo_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        found_quads = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 100:
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if not (len(approx) == 4 and cv2.isContourConvex(approx)):
                continue
            
            # Упорядочиваем точки для гомографии
            points = approx.reshape(4, 2)
            ordered = self._order_points(points)
            
            found_quads.append({
                'contour': approx,              # Оригинал для отрисовки (4, 1, 2)
                'area': area,
                'original_contour': contour,
                'corners': ordered              # Упорядоченные для гомографии (4, 2)
            })
            
        self.photo_binary = photo_binary
        self.found_quads = found_quads
        return contours

    # Step 2
    def _validate_candidates(self):
        """
        Шаг 2: Детекция нужного маркера ArUco.
        
        1. Вычисляет гомографию для каждого кандидата
        2. Нормализует изображение маркера
        3. Сравнивает с эталонной матрицей
        4. Возвращает найденный маркер с данными
        """
        
        for quad in self.found_quads:
            points = np.array(quad['corners'])
            
            # Обрезка (для ускорения обратнойгомографии)
            mins = points.min(axis=0).astype(int)
            maxs = points.max(axis=0).astype(int)

            binary_img_cropped = self.photo_binary[mins[1]:maxs[1]+1, mins[0]:maxs[0]+1]
            points -= mins
            
            # Сжатие (для больших ближних макреров)
            binary_img_compressed = cv2.resize(binary_img_cropped, (64, 64), interpolation=cv2.INTER_LINEAR)
            points *= 64.0 / np.array(binary_img_cropped.shape[::-1])
            
            # Нормализация
            cell_size = 10
            w = cell_size * self.marker.size
            dst_points = np.array([
                [0, 0], 
                [w-1, 0],
                [w-1, w-1],
                [0, w-1]
            ], dtype=np.float32)
            homography = cv2.getPerspectiveTransform(
                points.astype(np.float32), 
                dst_points
            )
            if homography is None or abs(np.linalg.det(homography)) < 1e-10:
                continue
            pattern_normalized = cv2.warpPerspective(binary_img_compressed, homography, (w, w))
            
            # Сжатие
            pattern_adjusted = cv2.resize(
                pattern_normalized, 
                (self.marker.size, self.marker.size), 
                interpolation=cv2.INTER_LINEAR
            )
            pattern = (pattern_adjusted > 127).astype(np.uint8)

            if self.marker.is_valid(pattern):
                return quad
        return None
    
    # Step 3
    def _refine_marker_corners(self, detected_marker):
        """
        Шаг 3: Субпиксельное вычисление углов маркера.
        
        1. Для каждой стороны находит уравнение прямой (метод наименьших квадратов)
        2. Вычисляет точки пересечения соседних сторон
        3. Создаёт фрейм вокруг маркера
        """
        
        original_contour = detected_marker['original_contour']
        corners_approx = detected_marker['corners']
        
        sides_points = self._split_contour_to_sides(original_contour, corners_approx)
        
        # Уравнения прямой для каждой стороны
        lines = []
        for side_points in sides_points:
            if len(side_points) >= 2:
                line = self._fit_line_least_squares(side_points)
                lines.append(line)
            else:
                # Если точек мало, используем линию между углами
                lines.append(None)
        
        # Пересечения соседних линий
        subpixel_corners = []
        for i in range(4):
            line1 = lines[i]
            line2 = lines[(i + 1) % 4] 
            
            if line1 is not None and line2 is not None:
                intersection = self._line_intersection(line1, line2)
                if intersection is not None:
                    subpixel_corners.append(intersection)
                else:
                    subpixel_corners.append(corners_approx[i])
            else:
                subpixel_corners.append(corners_approx[i])
        
        subpixel_corners = np.array(subpixel_corners, dtype=np.float32)
        
        return subpixel_corners


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Auxiliary functions (TODO: выделить в класс-предок (или нет?))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
       
    def _order_points(self, points):
        """
        Упорядочивает 4 точки в порядке: TL, TR, BR, BL
        Использует сумму и разность координат (работает при любом повороте)
        """
        if len(points) != 4:
            return points
        
        # Сумма координат: TL минимальная, BR максимальная
        s = points.sum(axis=1)
        tl = points[np.argmin(s)]
        br = points[np.argmax(s)]
        
        # Разность координат: TR минимальная, BL максимальная
        diff = np.diff(points, axis=1).flatten()
        tr = points[np.argmin(diff)]
        bl = points[np.argmax(diff)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _split_contour_to_sides(self, contour, corners):
        """
        Разделяет контур на 4 стороны на основе 4 углов.
        """
        contour_points = contour.reshape(-1, 2)
        sides = [[] for _ in range(4)]
        
        # Для каждой точки контура определяем к какой стороне она ближе
        for point in contour_points:
            min_dist = float('inf')
            closest_side = 0
            
            for i in range(4):
                pt1 = corners[i]
                pt2 = corners[(i + 1) % 4]
                
                # Расстояние от точки до отрезка
                dist = self._point_to_line_distance(point, pt1, pt2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_side = i
            
            sides[closest_side].append(point)
        
        return [np.array(side) for side in sides]

    def _point_to_line_distance(self, point, line_start, line_end):
        """Вычисляет расстояние от точки до отрезка."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Вектор отрезка
        dx, dy = x2 - x1, y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Параметр проекции
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        
        # Ближайшая точка на отрезке
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)

    def _fit_line_least_squares(self, points):
        """
        Аппроксимирует точки прямой методом наименьших квадратов.
        Возвращает коэффициенты (a, b, c) для уравнения ax + by + c = 0
        """
        if len(points) < 2:
            return None
        
        # Центрируем данные
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])
        
        x_centered = points[:, 0] - x_mean
        y_centered = points[:, 1] - y_mean
        
        # Ковариационная матрица
        cov_matrix = np.array([
            [np.sum(x_centered * x_centered), np.sum(x_centered * y_centered)],
            [np.sum(x_centered * y_centered), np.sum(y_centered * y_centered)]
        ])
        
        # Собственные значения и векторы
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Направляющий вектор - собственный вектор с наибольшим собственным значением
        direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Нормальный вектор (перпендикулярный направлению)
        normal = np.array([-direction[1], direction[0]])
        
        # Уравнение прямой: a*x + b*y + c = 0
        a, b = normal
        c = -(a * x_mean + b * y_mean)
        
        # Нормализуем
        norm = np.sqrt(a**2 + b**2)
        if norm > 0:
            a, b, c = a/norm, b/norm, c/norm
        
        return (a, b, c)

    def _line_intersection(self, line1, line2):
        """
        Вычисляет точку пересечения двух прямых.
        line = (a, b, c) для уравнения ax + by + c = 0
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        # Определитель
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:  # Прямые параллельны
            return None
        
        # Точка пересечения
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        
        return np.array([x, y], dtype=np.float32)

    def _debug_draw_ordered_corners(self, binary_bgr, selected_quads):
        """
        Отладочная функция: рисует упорядоченные угловые точки для каждого 4угольника.
        
        Цвета точек:
        1 (TL) - Красный (0, 0, 255)
        2 (TR) - Оранжевый (0, 140, 255)
        3 (BR) - Жёлтый (0, 255, 255)
        4 (BL) - Белый (255, 255, 255)
        
        Возвращает: изображение с нарисованными точками
        """
        debug_img = binary_bgr.copy()
        
        colors = [
            (0, 0, 255),      # 1: Красный (TL)
            (0, 140, 255),    # 2: Оранжевый (TR)
            (0, 255, 255),    # 3: Жёлтый (BR)
            (100, 100, 100)   # 4: Серый (BL)
        ]
        
        for idx, quad in enumerate(selected_quads):
            # Получаем точки контура и упорядочиваем их
            contour = quad['contour']
            points = contour.reshape(4, 2)
            ordered_points = self._order_points(points)
            
            # Рисуем 4 упорядоченные точки разными цветами
            for i, pt in enumerate(ordered_points):
                color = colors[i]
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 6, color, -1)
                
                # Добавляем номер точки для наглядности
                cv2.putText(debug_img, str(i+1), 
                        (int(pt[0]) - 8, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Опционально: рисуем номер квада рядом с первой точкой
            first_pt = ordered_points[0]
            cv2.putText(debug_img, f"Q{idx}", 
                    (int(first_pt[0]) + 15, int(first_pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_img

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _detect_candidates(self):
        start_time = time.perf_counter()
        img_edges = self._detect_and_filter_quads()
        end_time = time.perf_counter()
        self.log[-1]['1_detection_filter'] = end_time - start_time
        
        binary_bgr = cv2.cvtColor(self.photo_binary, cv2.COLOR_GRAY2BGR)
        self._save_image("1.1.binarization.jpg", binary_bgr)
        
        img_edges_viz = np.zeros_like(binary_bgr)
        cv2.drawContours(img_edges_viz, img_edges, -1, (0, 255, 0), 1)
        self._save_image("1.2.edges.jpg", img_edges_viz)

        img_selected_quads = self.photo_cropped.copy()
        for q in self.found_quads:
            cv2.drawContours(img_selected_quads, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image("1.3.selected_quads.jpg", img_selected_quads)
        
        debug_corners_img = self._debug_draw_ordered_corners(binary_bgr, self.found_quads)
        self._save_image("1.4.debug_ordered_corners.jpg", debug_corners_img)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =========================================================

if __name__ == "__main__":
    from pprint import pprint
    
    finder = ArucoFinder()
    
    photo = cv2.imread("../IMAGES_TEST/medium.jpg")
    if photo is None:
        raise RuntimeError("Ошибка: не удалось загрузить изображение")
    marker = Aruco(101, 6, cv2.aruco.DICT_6X6_250)
    
    results = finder.process(photo, marker)
    pprint(finder.log)
