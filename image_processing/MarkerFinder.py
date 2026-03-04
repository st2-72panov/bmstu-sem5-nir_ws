import cv2
import numpy as np
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")

class MarkerFinder:
    def __init__(self, marker_id, marker_dictionary):
        # self.frame contains coordinates of opposite diagonal points: ((x1, y1), (x2, y2))
        # By default (0, 0) and (0, 0), will be updated to image size in process if not set
        self.frame = None
        self.log = {}
        self.iteration_count = 0
        # TODO: calculate valid binary matrix for given marker
        

    def _create_output_dir(self):
        """Creates the output directory based on timestamp and iteration count."""
        self.iteration_count += 1
        now = datetime.now()
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{OUTPUT_DIR}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def _save_image(self, dir_path, filename, image):
        """Helper to save an image to the output directory."""
        path = os.path.join(dir_path, filename)
        cv2.imwrite(path, image)

    def _step1_prepare_images(self, photo):
        """
        Step 1: Add noise, create framed original AND cropped version.
        Returns: framed_original_img, cropped_img, log_time
        """
        start_time = time.perf_counter()
        
        h, w = photo.shape[:2]
        
        if self.frame == None:
            x1, y1 = 0, 0
            x2, y2 = w, h
            self.frame = ((x1, y1), (x2, y2))
        else:
            x1, y1 = self.frame[0]
            x2, y2 = self.frame[1]

        x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
        y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])

        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        noisy = np.clip(photo + noise, 0, 255).astype(np.uint8)

        framed_original = noisy.copy()
        cv2.rectangle(framed_original, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped = noisy[y1:y2, x1:x2]

        end_time = time.perf_counter()
        return framed_original, cropped, end_time - start_time

    def _step2_detect_and_filter_quads(self, image):
        """
        Step 2: Binarization, Quad Detection, and Filtering (Merged).
        Returns: binary_img, contours, selected_quads, log_time
        """
        start_time = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         
        found_quads = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 100:
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if not (len(approx) == 4 and cv2.isContourConvex(approx)):
                continue
            
            found_quads.append({
                'contour': approx,
                'area': area,
                'original_contour': contour
            })

        end_time = time.perf_counter()
        return binary, contours, found_quads, end_time - start_time

    def process(self, photo):
        """Main processing function."""
        self.log = {}
        output_dir = self._create_output_dir()

        framed_original, cropped_img, time_step1 = self._step1_prepare_images(photo)
        self.log['1_crop_noise_frame'] = time_step1
        self._save_image(output_dir, "1_crop_noise_frame.jpg", framed_original)

        binary_img, edges_img, selected_quads, time_step2 = self._step2_detect_and_filter_quads(cropped_img)
        self.log['2_detection_filter'] = time_step2
        
        binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        self._save_image(output_dir, "2_binarization.jpg", binary_bgr)
        
        edges_viz = np.zeros_like(binary_bgr)
        cv2.drawContours(edges_viz, edges_img, -1, (0, 255, 0), 1)
        self._save_image(output_dir, "3_edges.jpg", edges_viz)

        img_selected = cropped_img.copy()
        for q in selected_quads:
            cv2.drawContours(img_selected, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4_selected_quads.jpg", img_selected)

        return selected_quads

if __name__ == "__main__":
    finder = MarkerFinder()
    image = cv2.imread("../IMAGES_TEST/medium.jpg")
    marker_id = 101
    marker_dictionary = cv2.aruco.DICT_6X6_250
    if image is not None:
        results = finder.process(image, marker_id, marker_dictionary)
        print(finder.log)