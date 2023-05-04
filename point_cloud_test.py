#!/usr/bin/env python
import rospy
import ctypes
import struct
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError


def pc_callback(pointcloud):
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
    o3d.io.write_point_cloud("dataset/pointcloud.ply", pcd)


def listener():
    global num

    input("Press Enter to take an image")

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    #rospy.Subscriber("/zed_node/point_cloud/cloud_registered", PointCloud2, self.pointcloud_callback_10)
    msg = rospy.wait_for_message("/zed_node/point_cloud/cloud_registered", PointCloud2)

    pc_callback(msg)


def view_pointcloud():
    # Reading and visualizing the point cloud
    cloud = o3d.io.read_point_cloud("dataset/pointcloud.ply")
    o3d.visualization.draw_geometries([cloud])


if __name__ == '__main__':
    listener()
    view_pointcloud()