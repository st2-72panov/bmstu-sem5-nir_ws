from dataclasses import dataclass
from datetime import datetime
import os
import time
import logging
from contextlib import contextmanager

import cv2
import numpy as np

from Aruco import Aruco

"""
Именование изображений:
photo - оригинальный снимок, переданный в MarkerDetector
framed_... - (модифицированная) часть photo, находящаяся внутри frame*
img_... - разные изображения (для сохранения)
frame ((x_left_up, y_left_up), (x_right_bot, y_right_bot)) - рабочая область photo; предполагается, что маркер лежит внутри неё
"""

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_FOLDER = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")
FRAME_FACTOR = 2.0
KEYPOINTS_TO_FIND = 32
KEYPOINT_DISTANCE_THRESHOLD = 40
MATCHING_KEYPOINTS_MINIMUM = 5

class MarkerDetector:
    @dataclass
    class MarkerDetectorConfig:
        OUTPUT_DIR_FOLDER: str
    
    def __init__(self, reference_marker: Aruco):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = MarkerDetector.MarkerDetectorConfig(
            OUTPUT_DIR_FOLDER=os.path.join(script_dir, "IMAGES_OUTPUT")
        )
        
        self.logger = logging.getLogger(f"MarkerDetector_{id(self)}")
        self.timing_data = {}
        self.iteration_count = 0
        self.reference_marker = reference_marker
        # self.frame содержит координаты противоположных диагональных точек фрейма: ((x1, y1), (x2, y2))
        self.frame = None  # Frame - весь экран
        # self.frame = ((1280 // 4, 0), (1280 * 4 // 5, 720 * 2 // 3))
        # self.frame = ((0, 0), (100, 100))

        self.photo = None
        self.framed_photo = None
        self.framed_gray = None
        self.framed_binary = None

        self.FRAME_FACTOR = 2.0
        self.prev_quad = ...  # TODO: ?
        # TODO: сделать поворот, перемещение фрейма, адаптивный frame_factor
        
        self.prev_keypoints = self.prev_descriptors = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Утилиты логгирования и тайминга

    def _create_output_dir(self):
        now = datetime.now()
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{self.config.OUTPUT_DIR_FOLDER}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        self.output_dir = dir_name

    def _save_image(self, filename, image):
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, image)
    
    def _log_time(self, title: str, duration: float):
        """Запись времени выполнения шага"""
        self.timing_data[title] = duration
        self.logger.info(f"{title}: {duration:.4f}s")

    @contextmanager
    def timer(self, title: str):
        """Контекстный менеджер для замера времени выполнения блока кода"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self._log_time(title, duration)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Реализации шагов

    def _prepare_image(self, photo):
        """
        Шаг 1: Добавление шума, создание оригинала в рамке и обрезанной версии
        """
        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        self.photo = np.clip(photo + noise, 0, 255).astype(np.uint8)

        photo_with_frame = photo.copy()
        if self.frame is not None:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]
            cv2.rectangle(photo_with_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.framed_photo = photo[y1:y2, x1:x2]
        else:
            self.framed_photo = photo.copy()
        self._save_image("0.crop_noise_frame.jpg", photo_with_frame)
        
        self.framed_gray = cv2.cvtColor(self.framed_photo, cv2.COLOR_BGR2GRAY)

    
    def _find_and_match_keypoints(self):
        # 1. Поиск точек
        orb = cv2.ORB_create(
            # nfeatures=KEYPOINTS_TO_FIND,
            # scaleFactor=1.2,
            # nlevels=8,
            # edgeThreshold=31,
            # firstLevel=0,
            # WTA_K=2,
            # scoreType=cv2.ORB_FAST_SCORE,
            # patchSize=31,
            # fastThreshold=20
        )
        self.current_keypoints, self.current_descriptors = orb.detectAndCompute(self.framed_gray, None)
        
        # 2. Сравнение с точками предыдущего фото
        if self.prev_keypoints is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.current_descriptors, self.prev_descriptors)
        matches = [m for m in matches if m.distance <= KEYPOINT_DISTANCE_THRESHOLD]
        if len(matches) < MATCHING_KEYPOINTS_MINIMUM:
            return None
        
        # TODO: проверить работу; 
        #       рассмотреть возможность использования простого поворотно-масштабного преобразования
        # 3. Вычисление угловых точек
        H, _ = cv2.findHomography(self.prev_keypoints, self.current_keypoints, cv2.RANSAC, 5.0)
        if H is None:
            self.logger.warning('Неудачная гомография')
            return None
        corners = cv2.perspectiveTransform(self.prev_corners, H)
        return corners

    def _detect_candidates(self): pass
    def _validate_candidates(self): pass
    def _refine_marker_corners(self, detected_marker): pass
    def _frame_to_photo_coordinates(self, points: np.ndarray):
        if self.frame is None:
            return points
        return points + self.frame[0]  # TODO: проверить работу

    def _estimate_pose(self):
        ...  # TODO
        # TODO: провращать точки в каждом из методов субпиксельного уточнения

    def _calculate_next_frame(self):
        # Координаты следующего фрейма
        center = np.mean(self.subpixel_corners, axis=0)
        
        diag1 = np.linalg.norm(self.subpixel_corners[0] - self.subpixel_corners[2])
        diag2 = np.linalg.norm(self.subpixel_corners[1] - self.subpixel_corners[3])
        max_diag = max(diag1, diag2)
        frame_size = int(max_diag * self.FRAME_FACTOR)
        
        h, w = self.photo.shape[:2]
        x1 = max(0, int(center[0] - frame_size / 2))
        y1 = max(0, int(center[1] - frame_size / 2))
        x2 = min(w, int(center[0] + frame_size / 2))
        y2 = min(h, int(center[1] + frame_size / 2))
        
        self.frame = ((x1, y1), (x2, y2))

    def _save_keypoints_within_marker(self):
        # TODO: проверить, подходит ли subpixel_corners (np.ndarray ли это)
        # quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        mask = [cv2.pointPolygonTest(self.subpixel_corners, pt.pt, False) > 0 
                for pt in self.current_keypoints]
        self.prev_keypoints = [pt for pt, m in zip(self.current_keypoints, mask) if m]
        self.prev_descriptors = self.current_descriptors[mask]  # если descriptors — numpy массив
    
    def _debug_result_photo(self):
        # Фрейм
        (x1, y1), (x2, y2) = self.frame
        img = self.framed_photo.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Окружность + точка в центре
        for corner in self.subpixel_corners:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(img, (x, y), 8, (0, 0, 255), 1)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        
        # Особые точки
        img = cv2.drawKeypoints(
            img,
            self.prev_keypoints, 
            None,
            color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        self._save_image("4.subpixel_corners.jpg", img)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # TODO: merge steps 1 and 2; change places 4 and 5
    # TODO: по возможности убрать self. переменные, заменив их на локальные;
    #       рассмотреть замену subpix_corners на self. переменную
    def process(self, photo, isRepeatedAttempt=False):
        if not isRepeatedAttempt:
            self.iteration_count += 1
        self._create_output_dir()
        self.timing_data = {}
        self.logger.info(f"Начало обработки итерации #{self.iteration_count}")

        # .................................................
        with self.timer('0_prepare_image'):
            photo_with_frame = self._prepare_image(photo)

        # .................................................
        # Шаг 1: поиск и сопоставление особых точек, вычисление положения углов маркера
        with self.timer('1_find_and_match_keypoints'):
            corners = self._find_and_match_keypoints()
        # TODO: проверить работоспособность; робастность
        # TODO: визуализировать
        # () А если на пути маркера встанет помеха? Если какая-либо новая точка отнесётся к помехе 
        
        # .................................................
        if corners is not None:
            with self.timer('2.2A_subpixel_corners'):
                self._subpix_corners_by_keypoints(corners)  # TODO
        else:
            # .............................................
            with self.timer('1.1B_search_for_candidates'):
                self._detect_candidates()

            with self.timer('1.2B_validate_candidates'):
                detected_marker = self._validate_candidates()

            if detected_marker is None:
                if self.frame is not None:
                    self.frame = None
                    self.logger.warning("Маркер не найден, повторная попытка на полном изображении")
                    return self.process(photo, True)
                self.logger.warning("Маркер не найден")
                return None
            
            # .............................................
            with self.timer('2B_subpixel_corners'):
                self._refine_marker_corners(detected_marker)
                
        self.subpixel_corners = self._frame_to_photo_coordinates(self.subpixel_corners)

        # .................................................
        # with self.timer('3_estimate_pose'):
        #     pose = self._estimate_pose()

        # .................................................
        with self.timer('4_prepare_next_step'):
            self._save_keypoints_within_marker()
            self._calculate_next_frame()
        self._debug_result_photo()
        self.logger.info(f"Обработка завершена. Всего шагов: {len(self.timing_data)}")
        # return pose
