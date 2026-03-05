import cv2
import numpy as np

class Aruco:
    def __init__(self, id, size, dictionary):
        self.id = id
        self.size = size + 2  # TODO: may not work for greater scales?
        self.dictionary = dictionary
        self.pattern = self._get_pattern()
    
    def _get_pattern(self):
        """Генерирует бинарную матрицу маркера (size x size)."""
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        img = cv2.aruco.generateImageMarker(dictionary, self.id, self.size)
        return (img > 127).astype(np.uint8)

    def is_valid(self, pattern) -> bool:
        if pattern.shape != (self.size, self.size):
            return False
        
        # Проверяем все 4 поворота (против ч.с.)
        for rotation in range(4):
            rotated_pattern = np.rot90(self.pattern, k=rotation)
            if np.array_equal(pattern, rotated_pattern):  # TODO: возможно, ввести процентное сравнение (пр.: >90%)
                return True
        
        return False