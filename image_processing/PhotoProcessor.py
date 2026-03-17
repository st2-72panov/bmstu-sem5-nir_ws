import copy

import cv2

from Aruco import Aruco
from ArucoDetector import ArucoDetector

class PhotoProcessor:
    def __init__(self):
        ...
        reference_marker = Aruco(101, 6, cv2.aruco.DICT_6X6_250)
        self.marker_finder = ArucoDetector(reference_marker)
        self.pending_photo = None
        self.is_new_photo_there = False
    
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
        photo = cv2.imread("../IMAGES_TEST/0.jpg")
        self.callback(photo)
        pose1 = self.marker_finder.process(self.pending_photo)
        
        photo = cv2.imread("../IMAGES_TEST/1.jpg")
        self.callback(photo)
        pose2 = self.marker_finder.process(self.pending_photo)

if __name__ == "__main__":
    pp = PhotoProcessor()
    pp.main_loop()