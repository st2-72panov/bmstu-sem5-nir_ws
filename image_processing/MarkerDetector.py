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
frame - рабочая область photo; предполагается, что маркер лежит внутри неё
"""

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_FOLDER = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")
FRAME_FACTOR = 2.0

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
        self.iteration_count = 0
        self.reference_marker = reference_marker
        # self.frame содержит координаты противоположных диагональных точек фрейма: ((x1, y1), (x2, y2))
        self.frame = None  # Frame - весь экран
        # self.frame = ((1280 // 4, 0), (1280 * 4 // 5, 720 * 2 // 3))
        # self.frame = ((0, 0), (100, 100))

        self.photo = None
        self.framed_photo = None
        self.framed_binary = None

        self.FRAME_FACTOR = 2.0  # TODO: сделать адаптивный frame_factor на основе предыдущих координат
        self.prev_quad = ...
        self.timing_data = {}
        # TODO: сделать поворот фрейма?

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

        return photo_with_frame
    
    def _detect_candidates(self):
        pass

    def _validate_candidates(self):
        pass

    def _refine_marker_corners(self, detected_marker):
        pass

    def _calculate_next_frame(self, corners):
        # Координаты следующего фрейма
        center = np.mean(corners, axis=0)
        
        diag1 = np.linalg.norm(corners[0] - corners[2])
        diag2 = np.linalg.norm(corners[1] - corners[3])
        max_diag = max(diag1, diag2)
        
        frame_size = int(max_diag * self.FRAME_FACTOR)
        
        h, w = self.photo.shape[:2]
        x1 = max(0, int(center[0] - frame_size / 2))
        y1 = max(0, int(center[1] - frame_size / 2))
        x2 = min(w, int(center[0] + frame_size / 2))
        y2 = min(h, int(center[1] + frame_size / 2))
        
        next_frame = ((x1, y1), (x2, y2))
        
        # Изображение
        # Фрейм
        img_next_frame = self.framed_photo.copy()
        cv2.rectangle(img_next_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # # Границы маркера
        # for i in range(4):
        #     pt1 = (int(corners[i][0]), int(corners[i][1]))
        #     pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
        #     cv2.line(img_next_frame, pt1, pt2, (0, 255, 255), 2)
        
        # Окружность + точка в центре
        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(img_next_frame, (x, y), 8, (0, 0, 255), 1)
            cv2.circle(img_next_frame, (x, y), 1, (0, 0, 255), -1)
        
        return next_frame, img_next_frame

    def _estimate_pose(self, ordered_points):
        ...  # ?

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def process(self, photo, isRepeatedAttempt=False):
        if not isRepeatedAttempt:
            self.iteration_count += 1
        self._create_output_dir()
        self.timing_data = {}
        self.logger.info(f"Начало обработки итерации #{self.iteration_count}")

        # .................................................
        # Шаг 0
        with self.timer('0_crop_noise_frame'):
            photo_with_frame = self._prepare_image(photo)
        self._save_image("0.crop_noise_frame.jpg", photo_with_frame)

        # .................................................
        # Шаг 1: Поиск кандидатов
        with self.timer('1_detection_filter'):
            self._detect_candidates()

        # .................................................
        # Шаг 2: Валидация кандидатов
        with self.timer('2_marker_detection'):
            detected_marker = self._validate_candidates()

        if detected_marker is None:
            if self.frame is not None:
                self.frame = None
                self.logger.warning("Маркер не найден, повторная попытка на полном изображении")
                return self.process(photo, True)
            self.logger.warning("Маркер не найден")
            return None
        
        # .................................................
        # Шаг 3: Уточнение углов
        with self.timer('3_subpixel_corners'):
            subpixel_corners = self._refine_marker_corners(detected_marker)

        # .................................................
        # Шаг 4.1 Вычисление следующего фрейма
        with self.timer('4_calculate_next_frame'):
            frame_coords, framed_with_corners = self._calculate_next_frame(subpixel_corners)
        self.frame = frame_coords
        self._save_image("3.subpixel_corners.jpg", framed_with_corners)
        
        with self.timer('5_estimate_pose'):
            pose = self._estimate_pose(subpixel_corners)
        
        self.logger.info(f"Обработка завершена. Всего шагов: {len(self.timing_data)}")
        return pose
