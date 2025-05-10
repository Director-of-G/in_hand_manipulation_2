import rclpy
from rclpy.node import Node
import time

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')

        self.frequency = 10
        self.callback_count = 0

        # 创建一个定时器，每2秒触发一次
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

    def timer_callback(self):
        current_count = self.callback_count
        self.callback_count += 1
        self.get_logger().info(f"Callback {current_count} start!")

        # 模拟延时任务，延时3秒
        time.sleep(0.2)

        self.get_logger().info(f"Callback {current_count} finish!")

def main(args=None):
    rclpy.init(args=args)

    node = TimerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()