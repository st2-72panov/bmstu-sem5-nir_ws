import cv2
import numpy as np

class Aruco:
    def __init__(self, id, size, dictionary):
        self.id = id
        self.size = size + 2  # may not work for greater scales
        self.dictionary = dictionary
        self.pattern = self._get_pattern()
    
    def _get_pattern(self):
        """Генерирует бинарную матрицу маркера (size x size)."""
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        img = cv2.aruco.generateImageMarker(dictionary, self.id, self.size)
        return (img > 127).astype(np.uint8)

    def is_valid(self, pattern) -> int:
        """
        Проверяет соответствие предоставленного паттерна паттерну маркера.
        
        Args:
            pattern (with borders): np.array размера (size) x (size)
        
        Returns:
            int: количество поворотов против ч.с. (0, 1, 2, 3) если валиден
            None: если паттерн невалиден
        """
        # Проверка размера
        if pattern.shape != (self.size, self.size):
            return None
        
        # # Проверка чёрной границы (однопиксельная рамка должна быть чёрной)
        # if (np.any(pattern[0, :] != 0) or
        #     np.any(pattern[-1, :] != 0) or
        #     np.any(pattern[:, 0] != 0) or
        #     np.any(pattern[:, -1] != 0)):
        #     return None

        # Проверяем все 4 ориентации (повороты против ч.с.)
        for rotation in range(4):
            rotated_pattern = np.rot90(self.pattern, k=rotation)
            if np.array_equal(pattern, rotated_pattern):
                return rotation
        
        # Если ни одна ориентация не совпала
        return None