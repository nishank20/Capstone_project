
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
# This script listens /zed/zed_node/depth/depth_registered
# and calculates the average distance of the closest objects.
# Then, publishes average distance to shy_roboy/nearest_distance.

class ImageConverter:

    def __init__(self):
        self.depthtotal=0
        self.pctotal = 0
        self.imgtotal = 0
        # Initialize depth image listener and average distance publisher.
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed_node/depth/depth_registered", Image, self.callback_depth)
        self.rgb_sub = rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.callback_rgb)
        self.msg2 = rospy.Subscriber("/velodyne_points", PointCloud2, self.ros_to_velodyne_pcl)
        print("point_Cloud")
        print("point_Cloud")
        self.msg = rospy.Subscriber("/zed_node/point_cloud/cloud_registered", PointCloud2,self.ros_to_pcl)
        self.depth_image = []
        self.cv_image_rgb = []
        self.pointcloud = PointCloud2()
        self.callback_received = False

    def ros_to_pcl(self, pointcloud):
        """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

            Args:
                ros_cloud (PointCloud2): ROS PointCloud2 message

            Returns:
                pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
        """
        self.pctotal += 1
        transform = pc2.read_points(pointcloud, skip_nans=True)
        numpy_data = np.array(list(transform))
        np.save("dataset3/point_cloud/num_pointcloud_"+str(self.pctotal)+".npy",numpy_data)
    def ros_to_velodyne_pcl(self, pointcloud):
        """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

            Args:
                ros_cloud (PointCloud2): ROS PointCloud2 message

            Returns:
                pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
        """
        self.pctotal += 1
        transform = pc2.read_points(pointcloud, skip_nans=True)
        numpy_data = np.array(list(transform))
        np.save("dataset3/Velodyne_point_cloud/num_V_pointcloud_"+str(self.pctotal)+".npy",numpy_data)

    def callback_depth(self, data):
        try:
            self.depthtotal += 1
            self.depth_image=np.frombuffer(data.data,dtype=np.uint8).reshape(data.height,data.width, -1)
            gray = cv2.cvtColor(self.depth_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("dataset3/depth/depth_img_" +str(self.depthtotal) + ".jpeg",self.depth_image)
            self.callback_received = True
        except CvBridgeError as e:
            print(e)

    def callback_rgb(self, data):
        try:
            # Read rgb image.
            self.imgtotal +=1
            self.cv_image_rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            cv2.imwrite("dataset3/images/rgb_data_"+str(self.imgtotal)+".jpeg", self.cv_image_rgb)
        except CvBridgeError as e:
            print(e)

def main():
    ic = ImageConverter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.loginfo("Node running.")
    # 30 FPS = 60 Hz
    rate = rospy.Rate(10)
    try:
        # while not rospy.is_shutdown():
        #     rate.sleep()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()