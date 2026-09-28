#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

## @brief Detects the horizontal position of the line within a region of interest.
## @param roi A grayscale or BGR image strip (e.g. the bottom rows of a frame).
## @return The x-coordinate of the line's centroid within the ROI, or None
def detect_line_position(roi):
    # Convert to grayscale if it has 3 channels
    if len(roi.shape) == 3:
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = roi

    # Threshold found by trial and error in lab 2; line is darker than the floor
    _, bin = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # No contour found
    if not contours:
        return None

    # Assume the largest contour is the line
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)

    # No area found, skip the frame
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    return cx


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw',
                                                  self.callback, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        self.forward_speed = 0.2 
        self.k = 0.5 #steering   
        self.last_error = 0.0 

    def callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        #TODO fill in the details for processing an image to determine the position
        # of the line in the image and publishing a command to move the robot forward
        # while the line is in the center.

        frame_height, frame_width = cv_image.shape[:2]

        # Define ROI to be the bottom strip of the frame
        roi = cv_image[frame_height - 20:frame_height, :]
        cx = detect_line_position(roi)

        cmd = Twist()

        if cx is not None:
            # Error from -1 to +1, error is non 0 if cx is not in the center of the frame
            error = (cx - frame_width / 2) / (frame_width / 2)

            cmd.linear.x = self.forward_speed
            cmd.angular.z = -self.k * error 
            self.last_error = error
        else:
            # Turn to where it was last seen if line is lost
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5 if self.last_error >= 0 else 0.5

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    print("Starting...")
    rclpy.init(args=args)
    lf = LineFollower()
    try:
        rclpy.spin(lf)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv2.destroyAllWindows()
        lf.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
