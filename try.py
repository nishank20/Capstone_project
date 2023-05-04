import rospy
import math
import cv2
import numpy as np
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import rospy
import ctypes
import struct
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
# This script listens /zed/zed_node/depth/depth_registered
# and calculates the average distance of the closest objects.
# Then, publishes average distance to shy_roboy/nearest_distance.

class ImageConverter:
    def __init__(self):
        # Initialize depth image listener and average distance publisher.
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/zed_node/depth/depth_registered", Image, self.callback_depth)
        # self.rgb_sub = rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, self.callback_rgb)
        #self.rgb_sub = rospy.Subscriber("/zed_node/rgb/image_raw_color", Image, self.callback_rgb)
        print("point_Cloud")
        print("point_Cloud")
        self.msg = rospy.Subscriber("/velodyne_points", PointCloud2,self.pc_callback)
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
        transform = pc2.read_points(pointcloud, skip_nans=True)
        numpy_data = np.array(list(transform))
        np.save("num_pointcloud.npy",numpy_data)
    def pc_callback(self, pointcloud):
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        num = 0

        transform = pc2.read_points(pointcloud, skip_nans=True)
        data = list(transform)


        print(len(data))

        for x in data:
            num += 1
            print(num)

            test = x[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
            rgb = np.append(rgb, [[r, g, b]], axis=0)

        print("This line is executed")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float) / 255.0)
        o3d.io.write_point_cloud("dataset/pointcloud4.ply", pcd)

    def callback_depth(self, data):
        try:
            #print(data)
            # Read depth image.
            #self.depth_image = self.bridge.imgmsg_to_cv2(data)
            #self.depth_image = np.array(self.bridge.compressed_imgmsg_to_cv2(data))
            self.depth_image=np.frombuffer(data.data,dtype=np.uint8).reshape(data.height,data.width, -1)
            #cv2.normalize(self.depth_image, self.depth_image, 0, 1, cv2.NORM_MINMAX)
            self.callback_received = True
            cv2.imshow("Depth Image from my node", self.depth_image)
            cv2.waitKey(10)
            #cv2.destroyAllWindows()

        except CvBridgeError as e:
            print(e)

    def callbackDepth(self, data):
        print("all is well")
        try:
            NewImg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            depth_array = np.array(NewImg, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            print(data.encoding)
            print()
            cv2.imwrite("depth.png", depth_array * 255)
        except CvBridgeError as e:
            print(e)

    def Depthcallback(self, msg_depth):  # TODO still too noisy!
        try:
            # The depth image is a single-channel float32 image
            # the values is the distance in mm in z axis
            cv_image = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="32FC1")
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
            cv_image_array = np.array(cv_image, dtype=np.dtype('f8'))
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
            # Resize to the desired size
            cv_image_resized = cv2.resize(cv_image_norm, self.desired_shape, interpolation=cv2.INTER_CUBIC)
            self.depthimg = cv_image_resized
            cv2.imshow("Depth Image from my node", self.depthimg)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
    def callback_rgb(self, data):
        try:
            # Read rgb image.
            self.cv_image_rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            self.callback_received = True
            cv2.imshow("Image from my node", self.cv_image_rgb)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except CvBridgeError as e:
            print(e)

    def callback_pc(self, data):

        self.pointcloud = data

    def get_depth(self):
        # Show images.


        cv2.imshow('depth image', self.depth_image)
        cv2.waitKey(1)

def main():
    ic = ImageConverter()

    rospy.init_node('image_converter', anonymous=True)
    rospy.loginfo("Node running.")

    # 30 FPS = 60 Hz
    rate = rospy.Rate(60)

    try:
        while not rospy.is_shutdown():

            if ic.callback_received:
                ic.get_depth()

            rate.sleep()
    except KeyboardInterrupt:
        print ("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()