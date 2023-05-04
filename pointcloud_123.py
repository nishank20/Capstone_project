#!/usr/bin/env python
import ctypes
# from __future__ import print_function
# from roslib import message
# import time
# import sys
# import sensor_msgs.point_cloud2 as pc2
# from std_msgs.msg import String

import os
import struct

import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import open3d


import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sensor_msgs.point_cloud2 as pc2




class BackgroundRemover(object):

    def __init__(self, pre_smoothing=False, post_smoothing=False):
        # smoothing
        self.do_pre_smoothing = pre_smoothing
        self.do_post_smoothing = post_smoothing


class BackgroundRemover3(object):

    def __init__(self, pre_smoothing=False, post_smoothing=True):
        # smoothing
        self.do_pre_smoothing = pre_smoothing
        self.do_post_smoothing = post_smoothing


class GetData():
    def __init__(self, data_dir, object_name, unit_deg=5):
        # file name
        self.fname = os.path.join(data_dir, object_name)

        # Extension
        self.img_ext = ".jpeg"
        self.txt_ext = ".txt"
        self.pcl_ext = ".ply"

        # ROS
        self.update_timestamp = rospy.Time.now()
        rospy.logwarn("Before Subscribe")
        # rospy.Subscriber("/zed_node/left/image_rect_color", Image, self.image_callback_rgb_2)
        # rospy.Subscriber("/zed_node/left/image_rect_color", Image, self.image_callback_rgb_3)
        # rospy.Subscriber("/n2/kinect2/hd/image_depth_rect", Image, self.image_callback_depth_2)
        # rospy.Subscriber("/n3/kinect2/hd/image_depth_rect", Image, self.image_callback_depth_3)
        rospy.Subscriber("/zed_node/point_cloud/cloud_registered", PointCloud2, self.pointcloud_callback_10)
        #rospy.Subscriber("/zed_node/point_cloud/cloud_registered", PointCloud2, self.pointcloud_callback_3)

        # Done flag
        self.flag = {}
        self.init_flags(value=True)

        # opencv
        self.bridge = CvBridge()

        self.background_2 = BackgroundRemover()
        self.background_3 = BackgroundRemover3()

        # deg
        self.unit_deg = int(unit_deg)

        # Half
        num_i = 72
        half_i = int(num_i / 2)  # 36
        self.angle_set = []
        for x in range(half_i):
            self.angle_set.append(x)
            self.angle_set.append(x + half_i)
        self.i = self.angle_set[0]

    def init_flags(self, value=False):
        self.flag["rgb_2"] = value
        self.flag["rgb_3"] = value
        self.flag["depth_2"] = value
        self.flag["depth_3"] = value
        self.flag["pcl_2"] = value
        self.flag["pcl_3"] = value
        self.flag["bg_2"] = value
        self.flag["bg_3"] = value
        if (value == False):
            self.update_timestamp = rospy.Time.now()

    @staticmethod
    def deg2dxlunit(deg):
        return int(float(deg) * (4096. / 360.)) % 4096

    @staticmethod
    def dxlunit2deg(dxlunit):
        return int(360. * float(dxlunit) / 4096.)

    def get_filename(self, angle, postfix, extension):
        curr_deg = self.i * self.unit_deg
        return self.fname + "_angle" + str(angle) + "_" + str(curr_deg) + "deg_" + postfix + extension

    def image_callback_rgb_2(self, msg):
        """Save rgb (.jpeg) files"""
        # rospy.loginfo(self.update_timestamp)
        # rospy.loginfo(msg.header.stamp)
        # rospy.loginfo(msg.header.stamp > self.update_timestamp)
        if (not self.flag["rgb_2"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("RGB 2 start " + str(self.i))
            cv2_img_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("Received rgb 2 image! #", self.i)
            img_name = self.get_filename(2, "rgb", self.img_ext)
            # save
            cv2.imwrite(img_name, cv2_img_rgb)
            self.flag["rgb_2"] = True
            # chromakey
            self.background_2.remove_background(
                img_name,
                self.get_filename(2, "mask", self.img_ext),
                self.get_filename(2, "full", self.img_ext))
            print("rgb 2 data written! #{}".format(self.i))
            self.flag["bg_2"] = True

    def image_callback_rgb_3(self, msg):
        """Save rgb (.jpeg) files"""
        # rospy.loginfo(self.update_timestamp)
        # rospy.loginfo(msg.header.stamp)
        # rospy.loginfo(msg.header.stamp > self.update_timestamp)
        if (not self.flag["rgb_3"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("RGB 3 start " + str(self.i))
            cv2_img_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("Received rgb 3 image! #", self.i)
            img_name = self.get_filename(3, "rgb", self.img_ext)
            # save
            cv2.imwrite(img_name, cv2_img_rgb)
            self.flag["rgb_3"] = True
            # chromakey
            self.background_3.remove_background(
                img_name,
                self.get_filename(3, "mask", self.img_ext),
                self.get_filename(3, "full", self.img_ext))
            print("rgb 3 data written! #{}".format(self.i))
            self.flag["bg_3"] = True

    def image_callback_depth_2(self, msg):
        """Save depth (.txt) files"""
        if (not self.flag["depth_2"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("Depth 2 start " + str(self.i))
            print("depth data 2 processing 1 # {}".format(self.i))
            cv2_img_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            # cv2_img_depth_png = self.bridge.imgmsg_to_cv2(msg)
            print("depth data 2 processing 2 # {}".format(self.i))
            depth_array = np.array(cv2_img_depth, dtype=np.float32)
            print("depth data 2 processing 3 # {}".format(self.i))
            ###############################
            # IMAGE
            max_value = np.amax(depth_array)
            ratio = 255.0 / max_value
            gray_image = np.uint8(depth_array * ratio)
            cv2.imwrite(self.get_filename(2, "depth", self.img_ext), gray_image)
            ###############################
            # TEXT
            # [m] to [mm]
            depth_array /= 1000.0
            # This is fast.
            depth_array[depth_array == 0.0] = 'nan'
            # Save
            np.savetxt(self.get_filename(2, "depth", self.txt_ext), depth_array, fmt="%2.6f", newline='\n')
            # np.savetxt(self.get_filename("depth", self.txt_ext), depth_array/1000.0, fmt = "%2.6f", newline = '\n')
            print("depth data 2 written! # {}".format(self.i))
            self.flag["depth_2"] = True
            # cv2.imwrite("/home/sujong/dataset_ws/src/get_image_data/data/test_"+str(i)+".png", depth_array*255)

    def image_callback_depth_3(self, msg):
        """Save depth (.txt) files"""
        if (not self.flag["depth_3"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("Depth 3 start " + str(self.i))
            print("depth data 3 processing 1 # {}".format(self.i))
            cv2_img_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            # cv2_img_depth_png = self.bridge.imgmsg_to_cv2(msg)
            print("depth data 3 processing 2 # {}".format(self.i))
            depth_array = np.array(cv2_img_depth, dtype=np.float32)
            print("depth data 3 processing 3 # {}".format(self.i))
            ###############################
            # IMAGE
            max_value = np.amax(depth_array)
            ratio = 255.0 / max_value
            gray_image = np.uint8(depth_array * ratio)
            cv2.imwrite(self.get_filename(3, "depth", self.img_ext), gray_image)
            ###############################
            # TEXT
            # [m] to [mm]
            depth_array /= 1000.0
            # This is fast.
            depth_array[depth_array == 0.0] = 'nan'
            # Save
            np.savetxt(self.get_filename(3, "depth", self.txt_ext), depth_array, fmt="%2.6f", newline='\n')
            # np.savetxt(self.get_filename("depth", self.txt_ext), depth_array/1000.0, fmt = "%2.6f", newline = '\n')
            print("depth data 3 written! # {}".format(self.i))
            self.flag["depth_3"] = True
            # cv2.imwrite("/home/sujong/dataset_ws/src/get_image_data/data/test_"+str(i)+".png", depth_array*255)

    def thresholding(self, points_as_list):
        """Applies thresholding to given pcl list using ros params."""

        # Get Parameters
        pcl_threshold_xmin = rospy.get_param('pcl_threshold_x_min')
        pcl_threshold_xmax = rospy.get_param('pcl_threshold_x_max')
        pcl_threshold_ymin = rospy.get_param('pcl_threshold_y_min')
        pcl_threshold_ymax = rospy.get_param('pcl_threshold_y_max')
        pcl_threshold_zmin = rospy.get_param('pcl_threshold_z_min')
        pcl_threshold_zmax = rospy.get_param('pcl_threshold_z_max')
        RED_VALUE_THRESHOLD = 100
        GREEN_VALUE_THRESHOLD = 70
        BLUE_VALUE_THRESHOLD = 70
        result = list()
        for d in points_as_list:
            if d[0] > pcl_threshold_xmin and \
                    d[0] < pcl_threshold_xmax and \
                    d[1] > pcl_threshold_ymin and \
                    d[1] < pcl_threshold_ymax and \
                    d[2] > pcl_threshold_zmin and \
                    d[2] < pcl_threshold_zmax:

                # Colors
                color = struct.unpack('>BBBB', bytearray(struct.pack("f", d[3])))
                # argb = struct.unpack('<B', color[0])
                # print str(argb)

                if color[0] == 0 and color[1] == 0 and color[2] == 0:
                    pass
                elif color[2] >= RED_VALUE_THRESHOLD and \
                        color[1] <= GREEN_VALUE_THRESHOLD and \
                        color[0] <= BLUE_VALUE_THRESHOLD:
                    result.append((d[0], d[1], d[2]))

        return result
    def pointcloud_callback_2(self, msg):
        """Save point cloud (.ply) files"""
        if (not self.flag["pcl_2"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("PCL 2 start " + str(self.i))

            print("pointcloud data 2 processing 1 # {}".format(self.i))
            assert isinstance(msg, PointCloud2)
            print("pointcloud data 2 processing 2 # {}".format(self.i))
            cloud_points = point_cloud2.read_points(msg, field_names=("x", "y", "z","rgb"))
            print("pointcloud data 2 processing 3 # {}".format(self.i))
            data1=[]
            data2=[]
            for data in cloud_points:
                test = data[3]
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                pack = ctypes.c_uint32(i).value
                r = int((pack & 0x00FF0000) >> 16)
                g = int((pack & 0x0000FF00) >> 8)
                b = int((pack & 0x000000FF))
                data1.append([data[0], data[1], data[2]])
                data2.append([r,g,b])
            # Make all tuple to list
            # for j in range(len(cloud_points)):
            # 	cloud_points[j] = list(cloud_points[j])
            cloud_points1 = map(list, data1)
            cloud_color1=map(list,data2)

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud_points1)
            pcd.colors = open3d.utility.Vector3dVector(cloud_color1)

            print("pointcloud data 2 processing 4 # {}".format(self.i))

            open3d.io.write_point_cloud(self.get_filename(2, "pcl", self.pcl_ext), pcd, write_ascii=True)
            print("pointcloud data 2 written! # {}".format(self.i))
            self.flag["pcl_2"] = True

    def pointcloud_callback_5(self,msg):
            gen = point_cloud2.read_points(msg, skip_nans=True)
            int_data = list(gen)
            data2=[]
            data1=[]
            for x in int_data:
                test = x[3]
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint8(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                data1.append([x[0], x[1], x[2]])
                data2.append([r, g, b])

            cloud_points1 = map(list, data1)
            cloud_color1 = map(list, data2)

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud_points1)
            pcd.colors = open3d.utility.Vector3dVector(cloud_color1)

            print("pointcloud data 2 processing 4 # {}".format(self.i))

            open3d.io.write_point_cloud(self.get_filename(2, "pcl", self.pcl_ext), pcd, write_ascii=True)
            print("pointcloud data 2 written! # {}".format(self.i))
            self.flag["pcl_2"] = True

    def pointcloud_callback_6(self, mesh_msg):
        # initialize objects
        source_np = []
        source_color = []
        mesh_obj = open3d.geometry.PointCloud()

        # for each point, make numpy array
        for p in point_cloud2.read_points(mesh_msg, skip_nans=True):
            source_np.append([p[0], p[1], p[2]])
            test = p[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            source_color.append([r / 255, g / 255, b / 255])
            # print(source_color)

        # assign numpy array to Open3D object
        cloud_points1 = map(list, source_np)
        cloud_color1 = map(list, source_color)
        mesh_obj.points = open3d.utility.Vector3dVector(cloud_points1)
        mesh_obj.colors= open3d.utility.Vector3dVector(cloud_color1)

        print("pointcloud data 2 processing 4 # {}".format(self.i))

        open3d.io.write_point_cloud(self.get_filename(2, "pcl", self.pcl_ext), mesh_obj, write_ascii=True)
        print("pointcloud data 2 written! # {}".format(self.i))
        self.flag["pcl_2"] = True


    def pointcloud_callback_7(self, mesh_msg):
        print("in callback pointcloud_callback_7");
        xyz = []
        rgb = []
        gen = pc2.read_points(mesh_msg, skip_nans=True)
        int_data = list(gen)

        print(np.shape(int_data))

        for x in int_data:

            if x[2] < 1.5:
                xyz.append([x[0], x[1], x[2]])

                test = x[3]
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                rgb.append([r/255, g/255, b/255])
        xyznp = np.array(xyz)
        rgbnp=np.array(rgb)
        out_pcd = open3d.geometry.PointCloud()
        out_pcd.points = open3d.utility.Vector3dVector(xyznp)
        out_pcd.colors = open3d.utility.Vector3dVector(rgbnp)

        open3d.io.write_point_cloud(self.get_filename(2, "pcl", self.pcl_ext), out_pcd, write_ascii=True)
        print("pointcloud data 2 written! # {}".format(self.i))
        self.flag["pcl_2"] = True

    def pointcloud_callback_8(self, ros_point_cloud):

        print ("in callback unorg")
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        print (np.shape(int_data))

        for x in int_data:

            if x[2] < 1.5:
                xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)

                test = x[3]
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)

                rgb = np.append(rgb, [[r/255, g/255, b/255]], axis=0)

        out_pcd = open3d.geometry.PointCloud()
        out_pcd.points = open3d.utility.Vector3dVector(xyz)
        out_pcd.colors = open3d.utility.Vector3dVector(rgb)

        open3d.io.write_point_cloud("dataset/cloud.ply", out_pcd)

    def pointcloud_callback_9(self, ros_point_cloud):

        print ("in callback pointcloud_callback_9")

        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
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
            rgb = np.append(rgb, [[r/255, g, b]], axis=0)
        out_pcd = open3d.geometry.PointCloud()
        out_pcd.points = open3d.utility.Vector3dVector(xyz)
        out_pcd.colors = open3d.utility.Vector3dVector(rgb)

        open3d.io.write_point_cloud("dataset/cloud1.ply", out_pcd)
        self.flag["pcl_3"] = True

    def pointcloud_callback_10(self, pointcloud):
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
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb.astype(np.float) / 255.0)
        open3d.io.write_point_cloud("dataset/pointcloud" + str(self.i) +".ply", pcd)
        self.flag["pcl_3"] = True
    def pointcloud_callback_3(self, msg):
        """Save point cloud (.ply) files"""
        if (not self.flag["pcl_3"]) and (msg.header.stamp > self.update_timestamp):
            rospy.loginfo("PCL 3 start " + str(self.i))

            print("pointcloud data 3 processing 1 # {}".format(self.i))
            assert isinstance(msg, PointCloud2)
            print("pointcloud data 3 processing 2 # {}".format(self.i))
            cloud_points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z","rgb"), skip_nans=True))
            print("pointcloud data 3 processing 3 # {}".format(self.i))

            # Make all tuple to list
            # for j in range(len(cloud_points)):
            # 	cloud_points[j] = list(cloud_points[j])
            cloud_points = map(list, cloud_points)

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud_points)
            print("pointcloud data 3 processing 4 # {}".format(self.i))

            open3d.io.write_point_cloud(self.get_filename(3, "pcl", self.pcl_ext), pcd, write_ascii=True)
            print("pointcloud data 3 written! # {}".format(self.i))
            self.flag["pcl_3"] = True

    def is_all_done(self):
        return (False not in self.flag.values())

    # return (self.flag["rgb"] and self.flag["depth"] and self.flag["pcl"])

    def wait_for_target(self, motor, target):
        while not rospy.is_shutdown():
            rospy.sleep(0.2)
            try:
                pose = motor.get_present_position()[0]
                print("target: {} deg, pose: {} deg".format(
                    self.dxlunit2deg(target), self.dxlunit2deg(pose)))
                if abs(target - pose) < 10:
                    break
            except:
                pass

    def main(self):

        rospy.logwarn("Before the loop")
        rate = rospy.Rate(30)  # Hz

        # Loop
        counter = 0
        offset_deg = 2
        for x in self.angle_set:
            # unit_i
            self.i = x
            counter += 1
            print("\n# is {} [loop counter: {}]".format(self.i, counter))

            t_cur = rospy.get_time()


            # initialize done flag to (False)
            self.init_flags()

            # Wait for all done
            while not rospy.is_shutdown():
                if self.is_all_done():
                    break
                rate.sleep()

            dt = rospy.get_time() - t_cur
            rospy.logwarn("dt is {}".format(dt))
        print("Shooting is done!!!")
        rospy.signal_shutdown("Shooting is done!!!")


if __name__ == '__main__':
    rospy.init_node('image_listener')
    # name of the object
    sujong = GetData("dataset/",
                     "test_file2")


    sujong.main()