import cv2
import numpy as np

class Aruco:
    def __init__(self, id, dictionary):
        self.id = id
        self.dictionary = dictionary
        self.size = self._get_size()
        self.pattern = self._get_pattern()
    
    def _get_size(self):
        """
        Получает размер маркера из словаря.
        Для DICT_NxN_250 размер внутренней части = N, с рамкой = N+2
        """
        dict_name = cv2.aruco.getPredefinedDictionary(self.dictionary)
        # Извлекаем размер из названия словаря (например, 6x6)
        # Для DICT_6X6_250 внутренняя часть 6x6, с рамкой 8x8
        if "6X6" in str(self.dictionary):
            return 6 + 2  # 8x8 с рамкой
        elif "5X5" in str(self.dictionary):
            return 5 + 2  # 7x7 с рамкой
        elif "7X7" in str(self.dictionary):
            return 7 + 2  # 9x9 с рамкой
        elif "4X4" in str(self.dictionary):
            return 4 + 2  # 6x6 с рамкой
        else:
            return 8  # По умолчанию
    
    def _get_pattern(self):
        """
        Получает эталонную бинарную матрицу для конкретного id.
        Возвращает матрицу размера size x size.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        
        # generateImageMarker возвращает grayscale изображение (1 канал)
        img = cv2.aruco.generateImageMarker(dictionary, self.id, self.size * 10)
        
        # Не конвертируем, т.к. изображение уже grayscale
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Разделяем на ячейки
        cell_size = binary.shape[0] // self.size
        pattern = np.zeros((self.size, self.size), dtype=np.uint8)
        
        for i in range(self.size):
            for j in range(self.size):
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                cell = binary[y1:y2, x1:x2]
                # Если больше половины пикселей белые - ячейка белая (1)
                pattern[i, j] = 1 if np.mean(cell) > 127 else 0
        
        return pattern
    