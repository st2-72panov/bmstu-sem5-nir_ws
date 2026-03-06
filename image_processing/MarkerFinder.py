from dataclasses import dataclass
from datetime import datetime
import os
import time

import cv2
import numpy as np

from Aruco import Aruco

class MarkerFinder:  # TODO: заменить на Detector

    @dataclass
    class MarkerFinderConfig:
        OUTPUT_DIR_FOLDER: str
        
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = MarkerFinder.MarkerFinderConfig(
            OUTPUT_DIR_FOLDER=os.path.join(script_dir, "IMAGES_OUTPUT")
        )
        
        self.log = []
        self.iteration_count = 0
        self.reference_marker = ...
        # self.frame содержит координаты противоположных диагональных точек фрейма: ((x1, y1), (x2, y2))
        self.frame = None  # Frame - весь экран
        # self.frame = ((1280 // 4, 0), (1280 * 4 // 5, 720 * 2 // 3))
        # self.frame = ((0, 0), (100, 100))
    
        self.photo = None
        self.photo_cropped = None  # TODO: rename to photo_framed
        self.photo_binary = None
    
        self.FRAME_FACTOR = 2.0  # TODO: сделать адаптивный frame_factor на основе предыдущих координат
        self.prev_quad = ...
        # TODO: сделать поворот фрейма?
    
    # =================================================
    
    def _create_output_dir(self) -> str:
        now = datetime.now()
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{self.OUTPUT_DIR_FOLDER}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        self.output_dir = dir_name

    def _save_image(self, filename, image):
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, image)

    # =================================================
    
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
        
        # Изображение с фреймом и углами
        photo_with_frames = self.photo_cropped.copy()
        cv2.rectangle(photo_with_frames, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for corner in corners:
            cv2.circle(photo_with_frames, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
        for i in range(4):
            pt1 = (int(corners[i][0]), int(corners[i][1]))
            pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
            cv2.line(photo_with_frames, pt1, pt2, (0, 255, 255), 2)
        
        return next_frame, photo_with_frames

    def _estimate_pose(self, ordered_points):
        ...  # ?
    
    # =================================================
    
    def _step0_prepare_images(self, photo):
        """
        Шаг 1: Добавление шума, создание оригинала в рамке и обрезанной версии.
        Возвращает: framed_original_img, cropped_img
        """
        
        # Add noise
        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        self.photo = np.clip(photo + noise, 0, 255).astype(np.uint8)

        photo_with_frame = photo.copy()
        if self.frame is not None:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]
            cv2.rectangle(photo_with_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.photo_cropped = photo[y1:y2, x1:x2]
        else:
            self.photo_cropped = photo.copy()

        return photo_with_frame
    
    def _step12_detect_marker():
        pass
    
    def _step3_refine_marker_corners():
        pass
    
    # =================================================
    
    def process(self, photo, marker: Aruco, isRepeatedAttempt=False):
        if not isRepeatedAttempt:
            self.iteration_count += 1
        self._create_output_dir()
        self.log.append(dict())
        self.marker = marker

        # . . . . . . . . . . . . . . . . . . . . . . . . .
        # Шаг 0
        
        start_time = time.perf_counter()
        photo, photo_with_frame, photo_cropped = self._step0_prepare_images(photo)
        end_time = time.perf_counter()
        self.log[-1]['0_crop_noise_frame'] = end_time - start_time
        
        self._save_image("0.crop_noise_frame.jpg", photo_with_frame)

        # . . . . . . . . . . . . . . . . . . . . . . . . .
        # Шаги 1-2: Поиск и проверка кандидатов
        
        detected_marker = self._step12_detect_marker()
        
        if detected_marker is None:
            if self.frame is not None:  # Неудача при неполном фрейме -- попытка поиска в полном
                self.frame = None
                return self.process(photo, marker, True)
            return None
            
        # . . . . . . . . . . . . . . . . . . . . . . . . .
        # Шаг 3: Уточнение углов
          
        start_time = time.perf_counter()
        subpixel_corners = self._step3_refine_marker_corners(detected_marker)
        end_time = time.perf_counter()
        self.log[-1]['3_subpixel_corners'] = end_time - start_time
                
        frame_coords, framed_with_corners = self._calculate_next_frame(subpixel_corners, photo_cropped, photo.shape)
        self.frame = frame_coords
        self._save_image("3.subpixel_corners.jpg", framed_with_corners)
        
        return self._estimate_pose(subpixel_corners)
    