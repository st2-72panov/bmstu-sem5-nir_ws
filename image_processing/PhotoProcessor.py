import copy
import logging
import time

import cv2

from Aruco import Aruco
from ArucoDetector import ArucoDetector
from util.logging_config import setup_logging
from util.time_logger import TimeLogger

class PhotoProcessor:
    def __init__(self):
        ...
        reference_marker = Aruco(101, 6, cv2.aruco.DICT_6X6_250)
        self.marker_finder = ArucoDetector(reference_marker)
        self.pending_photo = None
        self.is_new_photo_there = False

        # Настройка логгера
        setup_logging()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.time_logger = TimeLogger(self.logger)
    
    def callback(self, photo):
        ...
        self.pending_photo = photo
        self.is_new_photo_there = True
    
    # def main_loop(self):
    #     while True:
    #         # TODO: sleep until is_new_photo_there
    #         photo = copy(self.pending_photo)  # ?
    #         current_pose = self.marker_finder.process(self.pending_photo)
    #         # TODO: send to topic

    def main_loop(self):
        for i in range(2):
            with self.time_logger.measure('', 'total'):
                photo = cv2.imread(f"../IMAGES_TEST/{i}.jpg")
                self.callback(photo)
                pose = self.marker_finder.process(self.pending_photo)
        
if __name__ == "__main__":
    pp = PhotoProcessor()
    pp.main_loop()
