import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

OUTPUT_DIR = "IMAGES_RAW"

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        # Подписка на топик, который создал мост
        self.subscription = self.create_subscription(
            Image,
            '/X3/camera',  # Убедитесь, что имя совпадает с тем, что в мосту
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.count = 0
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        self.get_logger().info('Node started. Waiting for images...')

    def listener_callback(self, msg):
        try:
            # Конвертация ROS сообщения в изображение OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Формирование имени файла
            filename = os.path.join(OUTPUT_DIR, f'image_{self.count:04d}.jpg')
            
            # Сохранение
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f'Saved: {filename}')
            
            self.count += 1
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()