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

def pc_callback(data):
    print("start of pc_callback");
    xyz = np.array([[0, 0, 0]])
    rgb = np.array([[0, 0, 0]])
    num = 0


    print("len:",len(data))

    for x in data:
        num += 1
        print("num",num)

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
    print("end of pc_callback");
    return pcd;
    # o3d.io.write_point_cloud("dataset/pointcloud.ply", pcd)

def load_point_clouds(voxel_size=0.0):
    print("start of load_point_clouds");
    pcds = []
    for i in range(1, 3):
        data = np.load("pcd_dataset/num_pointcloud_%d.npy" %i)
        pcd = pc_callback(data);
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    o3d.io.write_point_cloud("pcd_dataset/pointcloud.ply", pcds);
    print("end of load_point_clouds");
    return pcds;

def visualize_point_clouds():
    print("start of visualize_point_clouds");
    voxel_size = 0.02
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down,
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    print("end of visualize_point_clouds");




if __name__ == '__main__':
    visualize_point_clouds();