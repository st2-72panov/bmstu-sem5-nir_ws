import cv2
import numpy as np
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")
FRAME_FACTOR = 2.0  # Коэффициент размера фрейма

class MarkerFinder:
    def __init__(self, marker_id=101, marker_dictionary=cv2.aruco.DICT_6X6_250):
        # self.frame содержит координаты противоположных диагональных точек: ((x1, y1), (x2, y2))
        self.frame = None
        self.log = {}
        self.iteration_count = 0
        self.marker_id = marker_id
        self.marker_dictionary = marker_dictionary
        # Вычисляем размер маркера (внутренняя часть + рамка)
        self.marker_size = self._get_marker_size()
        # Получаем эталонную бинарную матрицу для маркера
        self.marker_pattern = self._get_marker_pattern()
    
    def _get_marker_size(self):
        """
        Получает размер маркера из словаря.
        Для DICT_NxN_250 размер внутренней части = N, с рамкой = N+2
        """
        dict_name = cv2.aruco.getPredefinedDictionary(self.marker_dictionary)
        # Извлекаем размер из названия словаря (например, 6x6)
        # Для DICT_6X6_250 внутренняя часть 6x6, с рамкой 8x8
        if "6X6" in str(self.marker_dictionary):
            return 6 + 2  # 8x8 с рамкой
        elif "5X5" in str(self.marker_dictionary):
            return 5 + 2  # 7x7 с рамкой
        elif "7X7" in str(self.marker_dictionary):
            return 7 + 2  # 9x9 с рамкой
        elif "4X4" in str(self.marker_dictionary):
            return 4 + 2  # 6x6 с рамкой
        else:
            return 8  # По умолчанию
    
    def _get_marker_pattern(self):
        """
        Получает эталонную бинарную матрицу для конкретного marker_id.
        Возвращает матрицу размера marker_size x marker_size.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(self.marker_dictionary)
        
        # generateImageMarker возвращает grayscale изображение (1 канал)
        marker_img = cv2.aruco.generateImageMarker(dictionary, self.marker_id, self.marker_size * 10)
        
        # Не конвертируем, т.к. изображение уже grayscale
        gray = marker_img if len(marker_img.shape) == 2 else cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Разделяем на ячейки
        cell_size = binary.shape[0] // self.marker_size
        pattern = np.zeros((self.marker_size, self.marker_size), dtype=np.uint8)
        
        for i in range(self.marker_size):
            for j in range(self.marker_size):
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                cell = binary[y1:y2, x1:x2]
                # Если больше половины пикселей белые - ячейка белая (1)
                pattern[i, j] = 1 if np.mean(cell) > 127 else 0
        
        return pattern
    
    def _create_output_dir(self):
        """Создаёт выходную директорию на основе временной метки и счётчика итераций."""
        self.iteration_count += 1
        now = datetime.now()
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{OUTPUT_DIR}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def _save_image(self, dir_path, filename, image):
        """Вспомогательная функция для сохранения изображения."""
        path = os.path.join(dir_path, filename)
        cv2.imwrite(path, image)

    def _step1_prepare_images(self, photo):
        """
        Шаг 1: Добавление шума, создание оригинала в рамке И обрезанной версии.
        Возвращает: framed_original_img, cropped_img, log_time
        """
        start_time = time.perf_counter()
        
        h, w = photo.shape[:2]
        
        if self.frame is None:
            x1, y1 = 0, 0
            x2, y2 = w, h
            self.frame = ((x1, y1), (x2, y2))
        else:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]

        x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
        y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])

        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        noisy = np.clip(photo + noise, 0, 255).astype(np.uint8)

        framed_original = noisy.copy()
        cv2.rectangle(framed_original, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped = noisy[y1:y2, x1:x2]

        end_time = time.perf_counter()
        return framed_original, cropped, end_time - start_time

    def _step2_detect_and_filter_quads(self, image):
        """
        Шаг 2: Бинаризация, обнаружение четырёхугольников и фильтрация (объединено).
        Возвращает: binary_img, contours, selected_quads, log_time
        """
        start_time = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        found_quads = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 100:
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if not (len(approx) == 4 and cv2.isContourConvex(approx)):
                continue
            
            found_quads.append({
                'contour': approx,
                'area': area,
                'original_contour': contour
            })

        end_time = time.perf_counter()
        return binary, contours, found_quads, end_time - start_time

    def _step3_detect_marker(self, cropped_img, selected_quads):
        """
        Шаг 3: Детекция нужного маркера ArUco.
        
        1. Вычисляет гомографию для каждого кандидата
        2. Нормализует изображение маркера
        3. Сравнивает с эталонной матрицей
        4. Возвращает найденный маркер с данными
        
        Возвращает: detected_marker, log_time
        detected_marker содержит: 'quad', 'homography', 'normalized_img', 'corners'
        """
        start_time = time.perf_counter()
        
        detected_marker = None
        best_score = 0
        
        for quad in selected_quads:
            contour = quad['contour']
            
            # Упорядочиваем точки: верхний-левый, верхний-правый, нижний-правый, нижний-левый
            points = contour.reshape(4, 2)
            ordered_points = self._order_points(points)
            
            # Целевые точки для нормализации (размер marker_size x marker_size)
            dst_points = np.array([
                [0, 0],
                [self.marker_size - 1, 0],
                [self.marker_size - 1, self.marker_size - 1],
                [0, self.marker_size - 1]
            ], dtype=np.float32)
            
            # Вычисляем гомографию
            homography, _ = cv2.findHomography(ordered_points.astype(np.float32), dst_points)
            
            if homography is None:
                continue
            
            # Нормализуем изображение маркера
            normalized_size = 200  # Размер для детального анализа
            normalized_img = cv2.warpPerspective(
                cropped_img, 
                homography, 
                (normalized_size, normalized_size)
            )
            
            # Преобразуем в бинарное изображение
            gray_norm = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
            _, binary_norm = cv2.threshold(gray_norm, 127, 255, cv2.THRESH_BINARY)
            
            # Разделяем на ячейки и сравниваем с паттерном
            cell_size = normalized_size // self.marker_size
            detected_pattern = np.zeros((self.marker_size, self.marker_size), dtype=np.uint8)
            
            for i in range(self.marker_size):
                for j in range(self.marker_size):
                    y1, y2 = i * cell_size, (i + 1) * cell_size
                    x1, x2 = j * cell_size, (j + 1) * cell_size
                    
                    # Берём центр ячейки для определения цвета
                    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                    detected_pattern[i, j] = 1 if binary_norm[cy, cx] > 127 else 0
            
            # Проверяем рамку (должна быть чёрной)
            border_valid = True
            for i in range(self.marker_size):
                if detected_pattern[0, i] != 0:  # Верхняя рамка
                    border_valid = False
                    break
                if detected_pattern[self.marker_size-1, i] != 0:  # Нижняя рамка
                    border_valid = False
                    break
                if detected_pattern[i, 0] != 0:  # Левая рамка
                    border_valid = False
                    break
                if detected_pattern[i, self.marker_size-1] != 0:  # Правая рамка
                    border_valid = False
                    break
            
            if not border_valid:
                continue
            
            # Сравниваем внутреннюю часть с эталоном
            inner_size = self.marker_size - 2
            inner_detected = detected_pattern[1:-1, 1:-1]
            inner_expected = self.marker_pattern[1:-1, 1:-1]
            
            match_count = np.sum(inner_detected == inner_expected)
            total_inner = inner_size * inner_size
            score = match_count / total_inner
            
            if score > best_score and score >= 0.8:  # Порог соответствия 80%
                best_score = score
                detected_marker = {
                    'quad': quad,
                    'homography': homography,
                    'normalized_img': normalized_img,
                    'corners': ordered_points,
                    'score': score,
                    'detected_pattern': detected_pattern
                }
        
        end_time = time.perf_counter()
        return detected_marker, end_time - start_time

    def _order_points(self, points):
        """
        Упорядочивает 4 точки в порядке: верхний-левый, верхний-правый, 
        нижний-правый, нижний-левый.
        """
        # Сортируем по y-координате
        y_sorted = points[np.argsort(points[:, 1])]
        
        # Первые две - верхние, последние две - нижние
        top = y_sorted[:2]
        bottom = y_sorted[2:]
        
        # Сортируем верхние по x
        top_left = top[np.argsort(top[:, 0])[0]]
        top_right = top[np.argsort(top[:, 0])[1]]
        
        # Сортируем нижние по x
        bottom_left = bottom[np.argsort(bottom[:, 0])[0]]
        bottom_right = bottom[np.argsort(bottom[:, 0])[1]]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _step4_subpixel_corners(self, cropped_img, detected_marker, image_shape):
        """
        Шаг 4: Субпиксельное вычисление углов маркера.
        
        1. Для каждой стороны находит уравнение прямой (метод наименьших квадратов)
        2. Вычисляет точки пересечения соседних сторон
        3. Создаёт фрейм вокруг маркера
        
        Возвращает: subpixel_corners, frame_coords, framed_image, log_time
        """
        start_time = time.perf_counter()
        
        # Получаем оригинальный контур для более точной аппроксимации
        original_contour = detected_marker['quad']['original_contour']
        corners_approx = detected_marker['corners']
        
        # Разделяем контур на 4 стороны
        sides_points = self._split_contour_to_sides(original_contour, corners_approx)
        
        # Для каждой стороны находим уравнение прямой
        lines = []
        for side_points in sides_points:
            if len(side_points) >= 2:
                line = self._fit_line_least_squares(side_points)
                lines.append(line)
            else:
                # Если точек мало, используем линию между углами
                lines.append(None)
        
        # Вычисляем пересечения соседних линий
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
        
        # Вычисляем центр четырёхугольника
        center = np.mean(subpixel_corners, axis=0)
        
        # Вычисляем диагонали
        diag1 = np.linalg.norm(subpixel_corners[0] - subpixel_corners[2])
        diag2 = np.linalg.norm(subpixel_corners[1] - subpixel_corners[3])
        max_diag = max(diag1, diag2)
        
        # Размер фрейма
        frame_size = int(max_diag * FRAME_FACTOR)
        
        # Вычисляем координаты фрейма (не выходя за границы)
        h, w = image_shape[:2]
        x1 = max(0, int(center[0] - frame_size / 2))
        y1 = max(0, int(center[1] - frame_size / 2))
        x2 = min(w, int(center[0] + frame_size / 2))
        y2 = min(h, int(center[1] + frame_size / 2))
        
        frame_coords = ((x1, y1), (x2, y2))
        
        # Создаём изображение с фреймом и углами
        framed_image = cropped_img.copy()
        
        # Рисуем фрейм
        cv2.rectangle(framed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Рисуем субпиксельные углы (красные круги)
        for corner in subpixel_corners:
            cv2.circle(framed_image, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
        
        # Рисуем линии сторон
        for i in range(4):
            pt1 = (int(subpixel_corners[i][0]), int(subpixel_corners[i][1]))
            pt2 = (int(subpixel_corners[(i + 1) % 4][0]), int(subpixel_corners[(i + 1) % 4][1]))
            cv2.line(framed_image, pt1, pt2, (0, 255, 255), 2)
        
        end_time = time.perf_counter()
        return subpixel_corners, frame_coords, framed_image, end_time - start_time

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

    def process(self, photo, marker_id=None, marker_dictionary=None):
        """Основная функция обработки."""
        # Обновляем параметры маркера если переданы
        if marker_id is not None:
            self.marker_id = marker_id
            self.marker_pattern = self._get_marker_pattern()
        if marker_dictionary is not None:
            self.marker_dictionary = marker_dictionary
            self.marker_size = self._get_marker_size()
            self.marker_pattern = self._get_marker_pattern()
        
        self.log = {}
        output_dir = self._create_output_dir()

        # Шаг 1
        framed_original, cropped_img, time_step1 = self._step1_prepare_images(photo)
        self.log['1_crop_noise_frame'] = time_step1
        self._save_image(output_dir, "1_crop_noise_frame.jpg", framed_original)

        # Шаг 2
        binary_img, edges_img, selected_quads, time_step2 = self._step2_detect_and_filter_quads(cropped_img)
        self.log['2_detection_filter'] = time_step2
        
        binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        self._save_image(output_dir, "2_binarization.jpg", binary_bgr)
        
        edges_viz = np.zeros_like(binary_bgr)
        cv2.drawContours(edges_viz, edges_img, -1, (0, 255, 0), 1)
        self._save_image(output_dir, "3_edges.jpg", edges_viz)

        img_selected = cropped_img.copy()
        for q in selected_quads:
            cv2.drawContours(img_selected, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4_selected_quads.jpg", img_selected)

        # Шаг 3: Детекция маркера
        detected_marker, time_step3 = self._step3_detect_marker(cropped_img, selected_quads)
        self.log['3_marker_detection'] = time_step3
        
        if detected_marker is not None:
            # Сохраняем нормализованное изображение маркера
            self._save_image(output_dir, "5_normalized_marker.jpg", detected_marker['normalized_img'])
            
            # Шаг 4: Субпиксельные углы
            subpixel_corners, frame_coords, framed_with_corners, time_step4 = self._step4_subpixel_corners(
                cropped_img, detected_marker, photo.shape
            )
            self.log['4_subpixel_corners'] = time_step4
            
            # Сохраняем изображение с углами
            self._save_image(output_dir, "6_subpixel_corners.jpg", framed_with_corners)
            
            # Обновляем фрейм для следующей итерации
            self.frame = frame_coords
            
            detected_marker['subpixel_corners'] = subpixel_corners
            detected_marker['frame_coords'] = frame_coords
            
            return detected_marker
        else:
            self.log['3_marker_detection'] = time_step3
            self.log['4_subpixel_corners'] = 0
            return None


if __name__ == "__main__":
    finder = MarkerFinder()
    image = cv2.imread("../IMAGES_TEST/medium.jpg")
    marker_id = 101
    marker_dictionary = cv2.aruco.DICT_6X6_250
    
    if image is not None:
        results = finder.process(image, marker_id, marker_dictionary)
        print(finder.log)
        if results is not None:
            print(f"Маркер найден! Score: {results['score']:.2f}")
            print(f"Субпиксельные углы: {results['subpixel_corners']}")
    else:
        print("Ошибка: не удалось загрузить изображение")
    