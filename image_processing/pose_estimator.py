import cv2
import numpy as np
import os
import time
from datetime import datetime

OUTPUT_DIR = "image_processing/IMAGES_OUTPUT"

class PoseEstimator:
    def __init__(self):
        # self.frame contains coordinates of opposite diagonal points: ((x1, y1), (x2, y2))
        # By default (0, 0) and (0, 0), will be updated to image size in process if not set
        self.frame = ((0, 0), (0, 0))
        self.log = {}
        self.iteration_count = 0

    def _create_output_dir(self):
        """Creates the output directory based on timestamp and iteration count."""
        self.iteration_count += 1
        now = datetime.now()
        # Replacing ':' with '-' for filesystem compatibility (Windows does not allow ':' in paths)
        timestamp = now.strftime("%d.%m_%H-%M-%S")
        dir_name = f"{OUTPUT_DIR}/{timestamp}_{self.iteration_count}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def _save_image(self, dir_path, filename, image):
        """Helper to save an image to the output directory."""
        path = os.path.join(dir_path, filename)
        cv2.imwrite(path, image)

    def _step1_crop_and_noise(self, photo):
        """
        Step 1: Crop by frame, add Gaussian noise, draw frame.
        Returns: processed_image, log_time
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

        # Crop
        cropped = photo[y1:y2, x1:x2].copy()

        # Add Gaussian Noise
        # Mean=0, Sigma=10 (small noise)
        noise = np.random.normal(0, 10, cropped.shape).astype(np.int16)
        noisy = np.clip(cropped + noise, 0, 255).astype(np.uint8)

        # Draw Frame (Green edges)
        # Frame relative to cropped image is now (0,0) to (w_crop, h_crop)
        # But the requirement says "photo with ... frame self.frame". 
        # Since we cropped, the frame borders are the image borders. 
        # To visualize the frame concept, we draw a rectangle at the borders.
        framed = cv2.rectangle(noisy, (0, 0), (noisy.shape[1]-1, noisy.shape[0]-1), (0, 255, 0), 2)

        end_time = time.perf_counter()
        return framed, end_time - start_time

    def _step2_detect_quads(self, image):
        """
        Step 2: Binarization (Otsu), Edges, Quad Detection (Ramer-Douglas).
        Returns: binary_img, edges_img, found_quads (list of contours), log_time
        """
        start_time = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Binarization (Otsu)
        # THRESH_OTSU finds optimal threshold value
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Edge Detection (Canny on binary or gray, using binary for strict boundaries)
        edges = cv2.Canny(binary, 100, 200)

        # 3. Detect Quadrilaterals
        # Find contours on binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        found_quads = []
        
        for cnt in contours:
            # Approximate contour using Ramer-Douglas-Peucker
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check if quadrilateral (4 vertices)
            if len(approx) == 4:
                # Check convexity (markers are usually convex)
                if cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if area > 100: # Filter tiny noise
                        found_quads.append({
                            'contour': approx,
                            'area': area,
                            'original_contour': cnt
                        })

        end_time = time.perf_counter()
        return binary, edges, found_quads, end_time - start_time

    def _step3_filter_quads(self, image, quads):
        """
        Step 3: Filter quads (Area < 0.5 img, Not inside another).
        Returns: selected_quads, log_time
        """
        start_time = time.perf_counter()

        img_area = image.shape[0] * image.shape[1]
        max_area = img_area * 0.5
        
        # Filter by area first
        valid_quads = [q for q in quads if q['area'] < max_area]
        
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
                    # FIX: Convert numpy array to tuple of Python integers
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

        end_time = time.perf_counter()
        return selected_quads, end_time - start_time

    def process(self, photo):
        """
        Main processing function.
        """
        # Reset log for this iteration
        self.log = {}
        output_dir = self._create_output_dir()

        # --- Step 1: Crop + Noise + Frame ---
        img_step1, time_step1 = self._step1_crop_and_noise(photo)
        self.log['1_crop_noise_frame'] = time_step1
        self._save_image(output_dir, "1_crop_noise_frame.jpg", img_step1)

        # --- Step 2: Detection (Binarization, Edges, Raw Quads) ---
        binary_img, edges_img, found_quads, time_step2 = self._step2_detect_quads(img_step1)
        self.log['2_detection_bin_edges'] = time_step2
        
        # Save 2) Binarization
        # Convert binary to BGR for saving consistency
        binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        self._save_image(output_dir, "2_binarization.jpg", binary_bgr)
        
        # Save 3) Edges
        edges_bgr = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)
        self._save_image(output_dir, "3_edges.jpg", edges_bgr)

        # Save 4.1) Found Quadrilaterals (before filtering)
        img_found = img_step1.copy()
        for q in found_quads:
            cv2.drawContours(img_found, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4.1_found_quads.jpg", img_found)

        # --- Step 3: Filtering (Area, Containment) ---
        selected_quads, time_step3 = self._step3_filter_quads(img_step1, found_quads)
        self.log['3_filter_quads'] = time_step3

        # Save 4.2) Selected Quadrilaterals
        img_selected = img_step1.copy()
        for q in selected_quads:
            cv2.drawContours(img_selected, [q['contour']], -1, (0, 255, 0), 2)
        self._save_image(output_dir, "4.2_selected_quads.jpg", img_selected)

        # Return selected quads (optional, but good practice)
        return selected_quads

# Example Usage (Commented out for script execution safety)
if __name__ == "__main__":
    finder = PoseEstimator()
    img = cv2.imread("./saved_images/3.png")
    if img is not None:
        results = finder.process(img)
        print(finder.log)
    pass