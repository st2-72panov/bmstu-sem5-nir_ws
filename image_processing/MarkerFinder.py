import cv2
import numpy as np
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "IMAGES_OUTPUT")

class MarkerFinder:
    def __init__(self):
        # self.frame contains coordinates of opposite diagonal points: ((x1, y1), (x2, y2))
        self.frame = ((1280 // 4, 0), (1280 * 4 // 5, 720 * 2 // 3))
        # self.frame = ((0, 0), (0, 0))
        self.log = {}
        self.iteration_count = 0

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
        
        # Handle default frame (0,0,0,0) -> full image
        x1, y1 = self.frame[0]
        x2, y2 = self.frame[1]
        
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            x1, y1 = 0, 0
            x2, y2 = w, h
            self.frame = ((x1, y1), (x2, y2))

        # Ensure coordinates are within bounds and ordered
        x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
        y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])

        # Add Gaussian Noise to original image
        noise = np.random.normal(0, 10, photo.shape).astype(np.int16)
        noisy = np.clip(photo + noise, 0, 255).astype(np.uint8)

        # Create Framed Original (for saving step 1)
        # Draw rectangle on the FULL noisy image to show crop area
        framed_original = noisy.copy()
        cv2.rectangle(framed_original, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create Cropped Version (for further processing, NO frame)
        cropped = noisy[y1:y2, x1:x2].copy()

        end_time = time.perf_counter()
        return framed_original, cropped, end_time - start_time

    def _step2_detect_and_filter_quads(self, image):
        """
        Step 2: Binarization, Edges, Quad Detection, and Filtering (Merged Step 2 & 3).
        Returns: binary_img, contours, found_quads, selected_quads, time_step2, time_step3
        """
        # --- Part 1: Detection (Original Step 2) ---
        start_time_step = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Binarization (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         
        # 3. Detect Quadrilaterals
        found_quads = []
        
        for contour in contours:
            # Approximate contour using Ramer-Douglas-Peucker
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if quadrilateral (4 vertices) & convex
            if not (len(approx) == 4 and cv2.isContourConvex(approx)):
                continue
            # Filter tiny noise
            area = cv2.contourArea(approx)
            if area > 100:
                found_quads.append({
                    'contour': approx,
                    'area': area,
                    'original_contour': contour
                })


        # --- Part 2: Filtering ---

        img_area = image.shape[0] * image.shape[1]
        max_area = img_area * 0.5
        
        # Filter by area first
        valid_quads = [q for q in found_quads if q['area'] < max_area]
        
        # Sort by area descending to help with containment check (larger first)
        valid_quads.sort(key=lambda k: k['area'], reverse=True)
        
        selected_quads = []
        
        # Check containment
        for i, q1 in enumerate(valid_quads):
            is_inside = False
            for j, q2 in enumerate(valid_quads):
                if i == j:
                    continue
                # q2 cannot contain q1 if it's smaller or equal
                if q2['area'] <= q1['area']:
                    continue
                
                all_points_inside = True
                for point in q1['contour']:
                    pt = (int(point[0][0]), int(point[0][1]))
                    dist = cv2.pointPolygonTest(q2['contour'], pt, False)
                    if dist < 0:
                        all_points_inside = False
                        break
                
                if all_points_inside:
                    is_inside = True
                    break
          
            if not is_inside:
                selected_quads.append(q1)

        end_time_step = time.perf_counter()
        time_step = end_time_step - start_time_step

        return binary, contours, found_quads, selected_quads, time_step

    def process(self, photo):
        """
        Main processing function.
        """
        # Reset log for this iteration
        self.log = {}
        output_dir = self._create_output_dir()

        # --- Step 1: Noise + Frame (Original) + Crop (for processing) ---
        framed_original, cropped_img, time_step1 = self._step1_prepare_images(photo)
        self.log['1_crop_noise_frame'] = time_step1
        # Save 1) Original with noise and green frame (UNCROPPED) 
        self._save_image(output_dir, "1_crop_noise_frame.jpg", framed_original)

        # Use cropped_img for all subsequent steps (NO frame visible)
        
        # --- Step 2: Detection + Filtering (Merged) ---
        binary_img, edges_img, found_quads, selected_quads, time_step2 = self._step2_detect_and_filter_quads(cropped_img)
        
        # Preserve log keys for functional compatibility
        self.log['2_detection_bin_edges'] = time_step2
        
        # Save 2) Binarization (CROPPED, no frame)
        binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        self._save_image(output_dir, "2_binarization.jpg", binary_bgr)
        
        # Save 3) Edges/Contours (CROPPED, no frame)
        edges_viz = np.zeros_like(binary_bgr)
        cv2.drawContours(edges_viz, edges_img, -1, (0, 255, 0), 1)
        self._save_image(output_dir, "3_edges.jpg", edges_viz)

        # Save 4.1) Found Quadrilaterals (CROPPED, no frame)
        img_found = cropped_img.copy()
        for q in found_quads:
            cv2.drawContours(img_found, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4.1_found_quads.jpg", img_found)

        # Save 4.2) Selected Quadrilaterals (CROPPED, no frame)
        img_selected = cropped_img.copy()
        for q in selected_quads:
            cv2.drawContours(img_selected, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4.2_selected_quads.jpg", img_selected)

        return selected_quads

if __name__ == "__main__":
    finder = MarkerFinder()
    image = cv2.imread("../IMAGES_TEST/medium.jpg")
    if image is not None:
        results = finder.process(image)
        print(finder.log)
