import cv2
import numpy as np

from detectors.QuadDetector import QuadDetector

class STagDetector(QuadDetector):
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

            # Выпрямление
            w = 256
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
            pattern_flat = cv2.warpPerspective(framed_binary_cropped, homography, (w, w))
            
            rotation = self.reference_marker.check(pattern_flat)
            if rotation is not None:
                quad['corners'] = [quad['corners'][(i - rotation + 4) % 4] for i in range(4)]
                return quad
        return None
