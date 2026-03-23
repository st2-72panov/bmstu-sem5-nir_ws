import cv2
import numpy as np

from detectors.QuadDetector import QuadDetector

class ArucoDetector(QuadDetector):
    def __init__(self, reference_marker, output_dir_folder):
        super().__init__(reference_marker, output_dir_folder)

    # Шаг 1.2 .............................................
    def _validate_candidates(self):
        """
        Шаг 1.2: Детекция нужного маркера ArUco.
        
        1. Вычисляет гомографию для каждого кандидата
        2. Нормализует изображение маркера
        3. Сравнивает с эталонной матрицей
        4. Возвращает найденный маркер с данными
        """
        
        for quad in self.found_quads:
            points = np.array(quad['corners'])
            
            # Обрезка (для ускорения обратной гомографии)
            mins = points.min(axis=0).astype(int)
            maxs = points.max(axis=0).astype(int)

            framed_binary_cropped = self.framed_binary[mins[1]:maxs[1]+1, mins[0]:maxs[0]+1]
            points -= mins
            
            # Сжатие (для больших ближних маркеров)
            framed_binary_compressed = cv2.resize(framed_binary_cropped, (64, 64), interpolation=cv2.INTER_LINEAR)
            points *= 64.0 / np.array(framed_binary_cropped.shape[::-1])
            
            # Выпрямление
            cell_size = 10
            w = cell_size * self.reference_marker.size
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
            pattern_flat = cv2.warpPerspective(framed_binary_compressed, homography, (w, w))
            
            # Сжатие
            pattern_adjusted = cv2.resize(
                pattern_flat, 
                (self.reference_marker.size, self.reference_marker.size), 
                interpolation=cv2.INTER_LINEAR
            )
            pattern = (pattern_adjusted > 127).astype(np.uint8)

            if self.reference_marker.is_valid(pattern):
                return quad
        return None
