import cv2
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

OUTPUT_DIR = ".IMAGES_RAW"
ONLY_IMAGE = True

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        
        self.subscription = self.create_subscription(
            Image,
            '/X3/camera',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.count = 0

        if not ONLY_IMAGE:
            OUTPUT_DIR += f'/{time.strftime('%H:%M:%S')}'
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        self.get_logger().info('Node started. Waiting for images...')

    def listener_callback(self, msg):
        if ONLY_IMAGE and self.count == 1:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if ONLY_IMAGE:
            filename = f'image_{time.strftime('%H:%M:%S')}.jpg'
        else:
            filename = f'{self.count}.jpg'
        filename = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filename, cv_image)
        self.get_logger().info(f'Saved: {filename}')
        self.image_saved = True
        
        if ONLY_IMAGE:    
            self.get_logger().info('First image saved. Shutting down...')
            rclpy.shutdown()
            

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
