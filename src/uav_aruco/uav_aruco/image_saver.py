# uav_aruco/image_saver.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
import time

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.declare_parameter('save_directory', '/tmp/uav_images')
        self.save_dir = self.get_parameter('save_directory').get_parameter_value().string_value
        
        os.makedirs(self.save_dir, exist_ok=True)

        # Топик моста GZ -> ROS
        self.subscription = self.create_subscription(Image, '/X3/camera', self.image_callback, 1)
        self.latest_image = None
        
        # Таймер 2 секунды для сохранения
        self.timer = self.create_timer(2.0, self.save_callback)

    def image_callback(self, msg):
        self.latest_image = msg

    def save_callback(self):
        if self.latest_image is not None:
            try:
                # Конвертация ROS Image -> OpenCV
                img = cv2.cvtColor(
                    cv2.imdecode(
                        cv2.imencode('.png', self.latest_image.data)[1], 
                        cv2.IMREAD_COLOR
                    ), cv2.COLOR_BGR2RGB
                )
                # Примечание: В зависимости от encoding в msg.data может быть сырой буфер.
                # Для простоты используем стандартный cv_bridge подход (подразумевается наличие cv_bridge)
                # Но строго по документации ros2.html тип sensor_msgs/msg/Image.
                # Ниже реализация через numpy для минимизации зависимостей, если cv_bridge нет, 
                # но обычно требуется cv_bridge для корректного decoding.
                # Используем cv2 напрямую на данных, предполагая стандартный encoding.
                
                filename = os.path.join(self.save_dir, f"image_{int(time.time())}.png")
                # Для корректной работы необходим cv_bridge, но в рамках задачи пишем логику сохранения
                from cv_bridge import CvBridge
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f'Изображение сохранено: {filename}')
            except Exception as e:
                self.get_logger().error(f'Ошибка сохранения: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()