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

class MarkerDetector:
    @dataclass
    class MarkerDetectorConfig:
        OUTPUT_DIR_FOLDER: str
        KEYPOINTS_TO_FIND: int = 32
        KEYPOINT_DISTANCE_THRESHOLD: int = 40
        MATCHING_KEYPOINTS_MINIMUM: int = 5
    
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
        self.logger.info(f"{title:<30}\t{duration:.4f}s")

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

    def _apply_noise(self, photo):
        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        photo = np.clip(photo + noise, 0, 255).astype(np.uint8)
        return photo

    def _prepare_image(self, photo):
        """
        Шаг 1: Добавление шума, создание оригинала в рамке и обрезанной версии
        """
        self.photo = photo.copy()
        photo_with_frame = photo.copy()
        self.framed_photo = photo.copy()
        if self.frame is not None:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]
            cv2.rectangle(photo_with_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.framed_photo = self.framed_photo[y1:y2, x1:x2]
        self._save_image("0.crop_noise_frame.jpg", photo_with_frame)
        
        self.framed_gray = cv2.cvtColor(self.framed_photo, cv2.COLOR_BGR2GRAY)

    
    def _find_and_match_keypoints(self):
        # BUG: очень кривая оценка угловых точек. Видимо, очень маленькая терпимость к шуму.
            # TODO: какие ещё части нуждаются в сглаживании? Сглаживание или бинаризация?

        # 1. Поиск точек
        orb = cv2.ORB_create()  # TODO: рассмотреть варианты аргументов
        self.current_keypoints, self.current_descriptors = orb.detectAndCompute(self.framed_gray, None)
        
        # 2. Сравнение с точками предыдущего фото
        if self.prev_keypoints is None or self.current_descriptors is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.prev_descriptors, self.current_descriptors)
        matches = [m for m in matches if m.distance <= self.config.KEYPOINT_DISTANCE_THRESHOLD]
        if len(matches) < self.config.MATCHING_KEYPOINTS_MINIMUM:
            return None
        
        # 3. Вычисление угловых точек
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # TODO: может, заранее переводить их в такую форму?
        curr_pts = np.float32([self.current_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 10.0)  # 10.0 — идеальный вручную подобранный коэффициент
        if H is None:
            self.logger.warning('Неудачная гомография [для примерного вычисления положения углов]')
            return None
        # Это работает даже с учётом того, что prev_corners в локальных координатах, потому что pts тоже в них
        prev_corners = self.prev_corners_local.reshape(-1, 1, 2).astype(np.float32)
        corners = cv2.perspectiveTransform(prev_corners, H)
        corners = corners.reshape(-1, 2)
        return corners

    def _subpix_corners_by_keypoints(self, corners):
        # TODO: оптимизировать (работает слишком медленно)
        #       учесть, что будет найден не тот угол?
        # BUG: может принять край за угол (видимо, если далеко от угла)
        # BUG: куда-то пропадает четвёртый угол
        corners = np.float32(corners)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (5, 5)
        zeroZone = (-1, -1)
        self.subpixel_corners = cv2.cornerSubPix(self.framed_gray, corners, winSize, zeroZone, criteria)
        
        # # DEBUG
        # img = self.framed_photo.copy()
        # for corner in self.subpixel_corners:
        #     x, y = int(corner[0]), int(corner[1])
        #     cv2.circle(img, (x, y), 8, (0, 0, 255), 1)
        #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        # self._save_image('2A.subpix_corners.png', img)
        # # /DEBUG
        
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
        mask = [cv2.pointPolygonTest(self.subpixel_corners, pt.pt, False) > 0 
                for pt in self.current_keypoints]
        self.prev_keypoints = [pt for pt, m in zip(self.current_keypoints, mask) if m]
        self.prev_descriptors = self.current_descriptors[mask]
    
    def _debug_result_photo(self):
        img = self.photo.copy()
        
        # Особые точки
        (x1, y1), (x2, y2) = self.prev_frame
        img[y1:y2, x1:x2] = cv2.drawKeypoints(
            self.framed_photo.copy(),
            self.prev_keypoints,
            None,
            color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Фрейм
        (x1, y1), (x2, y2) = self.frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Окружность + точка в центре
        for corners, color in ((self.subpixel_corners_global, (0, 0, 255)), (self.approx_corners_global, (255, 0, 0))):
            if corners is None: continue
            for corner in corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(img, (x, y), 8, color, 1)
                cv2.circle(img, (x, y), 1, color, -1)

        self._save_image("4.result.jpg", img)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # TODO: по возможности убрать self. переменные, заменив их на локальные
    def process(self, photo, isRepeatedAttempt=False):
        if not isRepeatedAttempt:
            self.iteration_count += 1
        self._create_output_dir()
        self.timing_data = {}
        self.logger.info(f"Iteration #{self.iteration_count}")

        # ...................................
        photo = self._apply_noise(photo)
        with self.timer('0\tprepare image'):
            self._prepare_image(photo)

        # ...................................
        # Шаг 1: поиск и сопоставление особых точек, вычисление положения углов маркера
        with self.timer('1\tfind and match keypoints'):
            corners = self._find_and_match_keypoints()
        # TODO: проверить работоспособность; робастность
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
            
            with self.timer('2a\tsubpixel corners'):
                self._subpix_corners_by_keypoints(corners)
        else:
            # ...............................
            with self.timer('1b.1\tdetect candidates'):
                self._detect_candidates()

            with self.timer('1b.2\tvalidate candidates'):
                detected_marker = self._validate_candidates()

            if detected_marker is None:
                if self.frame is not None:
                    self.frame = None
                    self.logger.warning("Маркер не найден, повторная попытка на полном изображении")
                    return self.process(photo, True)
                self.logger.warning("Маркер не найден")
                return None
            
            # ...............................
            with self.timer('2b\tsubpixel corners'):
                self._refine_marker_corners(detected_marker)
                
        self.prev_corners_local = self.subpixel_corners
        self.subpixel_corners_global = self._frame_to_photo_coordinates(self.subpixel_corners)
        
        # ...................................
        # with self.timer('3\testimate pose'):
        #     pose = self._estimate_pose()

        # ...................................
        with self.timer('4\tprepare next step'):
            self._save_keypoints_within_marker()
            self._calculate_next_frame()
        self._debug_result_photo()
        # return pose
