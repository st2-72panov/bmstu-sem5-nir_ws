import copy

import cv2

from ArucoFinder import ArucoFinder

class PhotoProcessor:
    def __init__(self):
        ...
        self.marker_finder = ArucoFinder()
        self.pending_photo = None
        self.is_new_photo_there = False
    
    def callback(self, photo):
        ...
        self.pending_photo = photo
        self.is_new_photo_there = True
    
    def main_loop(self):
        while True:
            # TODO: sleep until is_new_photo_there
            photo = copy(self.pending_photo)  # ?
            current_pose = self.marker_finder.process(self.pending_photo)
            # TODO: send to topic
