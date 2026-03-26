from dataclasses import dataclass
from datetime import datetime
import os
import logging

import cv2
import numpy as np

from markers.Aruco import Aruco
from util.time_logger import TimeLogger

"""
    Именование изображений:
    photo - оригинальный снимок, переданный в MarkerDetector
    framed_... - (модифицированная) часть photo, находящаяся внутри frame*
    img_... - разные изображения (для сохранения)
    frame ((x_left_up, y_left_up), (x_right_bot, y_right_bot)) - рабочая область photo; предполагается, что маркер лежит внутри неё
"""

class MarkerDetector:
    @dataclass
    class MarkerDetectorConfig:
        OUTPUT_DIR_FOLDER: str
        KEYPOINTS_TO_FIND: int = 32
        KEYPOINT_DISTANCE_THRESHOLD: int = 40
        KEYPOINTS_TO_MATCH: int = 10
        MATCHING_KEYPOINTS_MINIMUM: int = 5
        
    def __init__(self, reference_marker: Aruco, output_dir_folder: str):
        # Неизменная конфигурация детектора
        self.reference_marker = reference_marker
        self.config = MarkerDetector.MarkerDetectorConfig(
            OUTPUT_DIR_FOLDER=output_dir_folder
        )
        
        # Метаданные
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.time_logger = TimeLogger(self.logger)
        self.iteration_count = 0
        
        # Рабочие переменные
        self.photo = None
        self.framed_photo = None
        self.framed_gray = None
        self.framed_binary = None
        
        self.frame = None  # координаты противоположных диагональных точек фрейма: ((x1, y1), (x2, y2))
        # self.frame = ((1280 // 4, 0), (1280 * 4 // 5, 720 * 2 // 3))
        # self.frame = ((0, 0), (100, 100))

        self.FRAME_FACTOR = 5.0
        self.prev_quad = ...  # TODO: ?
        # TODO: сделать поворот, перемещение фрейма, адаптивный frame_factor
        
        self.prev_keypoints = self.prev_descriptors = None

    def _create_output_dir(self):
        now = datetime.now()
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{self.config.OUTPUT_DIR_FOLDER}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        self.output_dir = dir_name

    def _save_image(self, filename, image):
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, image)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Реализации шагов

    # Шаг 0 ...............................................
    def _apply_noise(self, photo):
        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        photo = np.clip(photo + noise, 0, 255).astype(np.uint8)
        return photo

    def _prepare_image(self, photo):
        """
        Шаг 1: Добавление шума, создание оригинала в рамке и обрезанной версии
        """
        photo = cv2.medianBlur(photo, 5)

        self.photo = photo.copy()
        photo_with_frame = photo.copy()
        self.framed_photo = photo.copy()
        if self.frame is not None:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]
            cv2.rectangle(photo_with_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.framed_photo = self.framed_photo[y1:y2, x1:x2]
        self._save_image("0.crop_noise_frame.jpg", photo_with_frame)
        
        self.prev_framed_gray = self.framed_gray
        self.framed_gray = cv2.cvtColor(self.framed_photo, cv2.COLOR_BGR2GRAY)

    # Шаг 1 ...............................................
    def _find_and_match_keypoints(self):
        # BUG: очень кривая оценка угловых точек. Видимо, очень маленькая терпимость к шуму.
            # 1) TODO: какие ещё части нуждаются в сглаживании? Сглаживание или бинаризация?
            # 2) TODO: ЛИБО отредактировать ORB детектор (область запоминания дескриптора)
            #           (потому что я вижу, что в этом есть потенциал: большая часть точек детектируется ИМЕННО внутри этого маркера)
            #          и потом надеятся, что почти все точки угадаются правильно
            #          ЛИБО написать программу, удаляющую несоответствующие точки
            # TODO: почему в одной и той же точке детектируется несколько особых точек?
        # 1. Поиск точек
        with self.time_logger.measure('1', 'keypoint search', 1):
            orb = cv2.ORB_create()  # TODO: рассмотреть варианты аргументов. Что вообще может повлиять на них? Освещённость? Шум? Летающие объекты в воздухе?
            self.current_keypoints, self.current_descriptors = orb.detectAndCompute(self.framed_gray, None)
        img_keypoins = self._draw_keypoints(self.framed_photo, self.current_keypoints)
        self._save_image('1.all_keypoints.png', img_keypoins)

        # 2. Сравнение с точками предыдущего фото
        with self.time_logger.measure('1', 'keypoint matching', 1):
            if self.prev_keypoints is None or self.current_descriptors is None:
                return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.prev_descriptors, self.current_descriptors)
            matches = [m for m in matches if m.distance <= self.config.KEYPOINT_DISTANCE_THRESHOLD]
            if len(matches) < self.config.MATCHING_KEYPOINTS_MINIMUM:
                return None
        
        # 3. Вычисление угловых точек
        with self.time_logger.measure('1', 'corners calculation', 1):
            # 3.1 Грубая гомография .......................
            with self.time_logger.measure('1', 'rough homography', 2):
                prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # TODO: может, заранее переводить их в такую форму?
                curr_pts = np.float32([self.current_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 10.0)  # 10.0 — идеальный вручную подобранный коэффициент
                if H is None:
                    self.logger.warning('Неудачная гомография [для примерного вычисления положения углов]')
                    return None

            # 3.2 Фильтрация худших точек .................
            with self.time_logger.measure('1', 'worst kp filtration', 2):
                projected_prev_pts = cv2.perspectiveTransform(prev_pts, H)

                # diff имеет размерность (N, 1, 2), norm считаем по последнему измерению (axis=2)
                diff = projected_prev_pts - curr_pts
                distances = np.linalg.norm(diff, axis=2).flatten()
    
                partition_idx = len(distances) // 2
                median_dist = np.partition(distances, partition_idx)[partition_idx]  # Альтернатива медленному np.median
                good_mask = distances <= median_dist
                good_prev_pts = prev_pts[good_mask]
                good_curr_pts = curr_pts[good_mask]

            # 3.3 Улучшенная гомография ...................
            with self.time_logger.measure('1', 'good homography', 2):
                if len(good_prev_pts) >= 4:
                    H, _ = cv2.findHomography(good_prev_pts, good_curr_pts, cv2.RANSAC, 2.0)
                    matches = [m for i, m in enumerate(matches) if good_mask[i]]

            # 3.4 Вычисление углов ........................
            with self.time_logger.measure('1', 'corners calculation', 2):
                # Это работает даже с учётом того, что prev_corners в локальных координатах, потому что pts тоже в них
                prev_corners = self.prev_corners_local.reshape(-1, 1, 2).astype(np.float32)
                corners = cv2.perspectiveTransform(prev_corners, H)
                corners = corners.reshape(-1, 2)

        # 4. Матчинг
        with self.time_logger.measure('1', 'render keypoint match img', 1):
            c = min(len(matches), self.config.KEYPOINTS_TO_MATCH)
            best_matches = sorted(matches, key=lambda m: m.distance)[:c]  # TODO: можно провести сортировку через distances
            self._render_keypoint_match_img(self.prev_framed_gray, self.framed_gray, best_matches)

        return corners
    
    # Шаг 1b ..............................................
    def _detect_candidates(self): pass
    def _validate_candidates(self): pass
    
    # Шаг 2 ...............................................
    def _refine_quad_corners(self, detected_marker): pass  # refine_quad_corners_with_line_intersections
    def _refine_corners(self, corners):  # _refine_corners_by_harris
        corners = np.float32(corners)
        ordered_corners = self._order_points(corners)
        w, h = ordered_corners[1][0] - ordered_corners[3][0], ordered_corners[2][1] - ordered_corners[0][1]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        winSize = (int(0.1 * w), int(0.1 * h))
        zeroZone = (-1, -1)
        return cv2.cornerSubPix(self.framed_gray, ordered_corners, winSize, zeroZone, criteria)
        
        # # DEBUG
        # img = self.framed_photo.copy()
        # for corner in self.subpixel_corners:
        #     x, y = int(corner[0]), int(corner[1])
        #     cv2.circle(img, (x, y), 8, (0, 0, 255), 1)
        #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        # self._save_image('2A.subpix_corners.png', img)
        # # /DEBUG

    # Шаг 3 ...............................................
    def _estimate_pose(self):
        ...  # TODO
        # TODO: провращать точки в каждом из методов субпиксельного уточнения

    # Шаг 4 ...............................................
    def _calculate_next_frame(self):
        # Координаты следующего фрейма
        center = np.mean(self.subpixel_corners_global, axis=0)
        
        diag1 = np.linalg.norm(self.subpixel_corners_global[0] - self.subpixel_corners_global[2])
        diag2 = np.linalg.norm(self.subpixel_corners_global[1] - self.subpixel_corners_global[3])
        max_diag = max(diag1, diag2)
        frame_size = int(max_diag * self.FRAME_FACTOR)
        
        h, w = self.photo.shape[:2]
        x1 = max(0, int(center[0] - frame_size / 2))
        y1 = max(0, int(center[1] - frame_size / 2))
        x2 = min(w, int(center[0] + frame_size / 2))
        y2 = min(h, int(center[1] + frame_size / 2))
        
        self.prev_frame = self.frame
        if self.prev_frame is None:
            height, width = self.photo.shape[:2]
            self.prev_frame = ((0, 0), (width, height))
        self.frame = ((x1, y1), (x2, y2))

    def _save_keypoints_within_marker(self):
        # Shrink marker for not to save black edge keypoints
        quad = self.subpixel_corners
        quad = np.array(self.rescale_quad(quad, 0.9))

        # Filter keypoints
        mask = [cv2.pointPolygonTest(quad, pt.pt, False) > 0 
                for pt in self.current_keypoints]
        self.prev_keypoints = [pt for pt, m in zip(self.current_keypoints, mask) if m]
        self.prev_descriptors = self.current_descriptors[mask]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Вспомогательные функции

    def _rescale_quad(self, quad, scale):  # Точки в quad должны следовать друг за другом по кругу
        return np.array([p + (scale - 1.0) * (p - quad[(i + 2) % 4]) for i, p in enumerate(quad)])

    def _frame_to_photo_coordinates(self, points: np.ndarray):
        if self.frame is None:
            return points
        return points + self.frame[0]
    

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Отрисовка изображений
    
    def _render_keypoint_match_img(self, prev_img, cur_img, matches):
        # Создание цветного изображения для визуализации
        h1, w1 = prev_img.shape[:2]
        h2, w2 = cur_img.shape[:2]
        img_matching = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        img_matching[:h1, :w1] = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)
        img_matching[:h2, w1:w1+w2] = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)

        # Отрисовка точек и линий с разными цветами
        for i, m in enumerate(matches):
            # Генерация цвета через HSV (OpenCV формат: H[0-180], S[0-255], V[0-255])
            hue = int(180 * i / len(matches)) if len(matches) > 1 else 60
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)  # Конвертация в кортеж int
            
            # Координаты точек
            pt1 = tuple(int(x) for x in self.prev_keypoints[m.queryIdx].pt)
            pt2 = (int(self.current_keypoints[m.trainIdx].pt[0] + w1), 
                int(self.current_keypoints[m.trainIdx].pt[1]))
            
            # Отрисовка
            cv2.line(img_matching, pt1, pt2, color, 1)
            cv2.circle(img_matching, pt1, 3, color, -1)
            cv2.circle(img_matching, pt2, 3, color, -1)
            
        self._save_image('1.matching_keypoints.png', img_matching)

    def _render_result_img(self):
        img = self.photo.copy()
        
        # # Особые точки
        # (x1, y1), (x2, y2) = self.prev_frame
        # img[y1:y2, x1:x2] = cv2.drawKeypoints(
        #     self.framed_photo.copy(),
        #     self.prev_keypoints,
        #     None,
        #     color=(0, 255, 0), 
        #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        # )
        
        # Фрейм
        (x1, y1), (x2, y2) = self.frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Окружность + точка в центре
        for corners, size, color in ((self.subpixel_corners_global, 8, (0, 0, 255)), (self.approx_corners_global, 10, (255, 0, 0))):
            if corners is None: continue
            for corner in corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(img, (x, y), size, color, 1)
                cv2.circle(img, (x, y), 1, color, -1)

        self._save_image("4.result.jpg", img)

    def _draw_keypoints(self, img0, keypoints):
        img = img0.copy()
        # Отрисовка точек и линий с разными цветами
        for i, pt in enumerate(keypoints):
            # Генерация цвета через HSV (OpenCV формат: H[0-180], S[0-255], V[0-255])
            hue = int(180 * i / len(keypoints)) if len(keypoints) > 1 else 60
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(int(c) for c in color_bgr)  # Конвертация в кортеж int
            
            # Координаты точек
            pt = tuple(int(x) for x in pt.pt)
            cv2.circle(img, pt, 3, color, -1)
        return img

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # TODO: убрать self. переменные, нужные лишь для следующей функции
    def process(self, photo, isRepeatedAttempt=False):
        if not isRepeatedAttempt:
            self.iteration_count += 1
        self._create_output_dir()
        self.logger.info(f"Iteration #{self.iteration_count}")

        # ...................................
        photo = self._apply_noise(photo)
        with self.time_logger.measure('0', 'prepare image'):
            self._prepare_image(photo)

        # ...................................
        # Шаг 1: поиск и сопоставление особых точек, вычисление положения углов маркера
        with self.time_logger.measure('1', 'find and match keypoints'):
            corners = self._find_and_match_keypoints()
        # TODO: проверить робастность
        #       () А если на пути маркера встанет помеха? Если какая-либо новая точка отнесётся к помехе 
        
        if corners is not None:
            self.approx_corners_global = self._frame_to_photo_coordinates(corners)
        else:
            self.approx_corners_global = None
        
        # ...................................
        if corners is not None:
            # # DEBUG
            # img = self.framed_photo.copy()
            # for corner in corners:
            #     x, y = int(corner[0]), int(corner[1])
            #     cv2.circle(img, (x, y), 8, (0, 0, 255), 1)
            #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            # self._save_image('1a.calculated_corners.png', img)
            # # /DEBUG
            
            with self.time_logger.measure('2a', 'subpixel corners'):
                self.subpixel_corners = self._refine_corners(corners)
        else:
            # ...............................
            with self.time_logger.measure('1b.1', 'detect candidates'):
                self._detect_candidates()

            with self.time_logger.measure('1b.2', 'validate candidates'):
                detected_marker = self._validate_candidates()

            if detected_marker is None:
                if self.frame is not None:
                    self.frame = None
                    self.logger.warning("Маркер не найден, повторная попытка на полном изображении")
                    return self.process(photo, True)
                self.logger.warning("Маркер не найден")
                return None
            
            # ...............................
            with self.time_logger.measure('2b', 'subpixel corners'):
                self._refine_quad_corners(detected_marker)
                
        self.prev_corners_local = self.subpixel_corners
        self.subpixel_corners_global = self._frame_to_photo_coordinates(self.subpixel_corners)
        
        # ...................................
        # with self.time_logger.measure('3', 'estimate pose'):
        #     pose = self._estimate_pose()

        # ...................................
        with self.time_logger.measure('4', 'prepare next step'):
            self._save_keypoints_within_marker()
            self._calculate_next_frame()
        self._render_result_img()
        # return pose
