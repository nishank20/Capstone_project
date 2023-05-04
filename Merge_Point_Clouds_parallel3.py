import os

import numpy as np
import open3d as o3d
import struct
import ctypes
from joblib import Parallel, delayed
import multiprocessing
import subprocess



voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

def pc_callback(data):
    print("start of pc_callback");
    xyz = np.array([[0, 0, 0]])
    rgb = np.array([[0, 0, 0]])
    num = 0


    print("len:",len(data))

    for x in data:
        num += 1
        # print("num",num)

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


# def load_point_clouds(voxel_size=0.0):
#     print("start of load_point_clouds", voxel_size);
#     pcds = []
#     lstOfFiles = os.listdir("pcd_dataset/Velodyne_pointcloud/");
#     for filename in lstOfFiles:
#         if filename.endswith('.npy'):
#             print("file name::", filename);
#             data = np.load("pcd_dataset/Velodyne_pointcloud/"+filename)
#             pcd = pc_callback(data);
#             pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
#             pcd_down.estimate_normals()
#             pcds.append(pcd_down)
#     print("end of pc_callback");
#     return pcds

def load_point_clouds(voxel_size=0.0):
    print("start of load_point_clouds", voxel_size);
    pcds = []
    for i in range(1,201):
        try:
            print("dataset/Velodyne_pointcloud/num_V_pointcloud_%d.npy" %i);
            data = np.load("dataset/Velodyne_pointcloud/num_V_pointcloud_%d.npy" %i)
            pcd = pc_callback(data);
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            pcd_down.estimate_normals()
            pcds.append(pcd_down)
        except:
            print("dataset/Velodyne_pointcloud/num_V_pointcloud_%d.npy not found" %i)
    print("end of pc_callback");
    return pcds

def pairwise_registration(source, target, source_id, target_id, odometry, pose_graph):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    print("Build o3d.registration.PoseGraph")
    if target_id == source_id + 1:  # odometry case
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                     target_id,
                                                     transformation_icp,
                                                     information_icp,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                     target_id,
                                                     transformation_icp,
                                                     information_icp,
                                                     uncertain=True))
    print("source_id::", source_id)
    print("end of pairwise_registration");
    return pose_graph;


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    print("start of full_registration");
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    MAX_THREAD = min(multiprocessing.cpu_count(),
                     max(len(pose_graph.edges), 1))
    # Parallel(n_jobs=MAX_THREAD)(Parallel(n_jobs=MAX_THREAD)(
    #     delayed(pairwise_registration)(pcds[int(source_id)], pcds[int(target_id)], int(source_id), int(target_id), odometry, pose_graph)
    #     for target_id in range(int(source_id) + 1, n_pcds)) for source_id in range(n_pcds))
    with Parallel(n_jobs=MAX_THREAD):
        for source_id in range(n_pcds) :
            Parallel(n_jobs=MAX_THREAD)(delayed(pairwise_registration)(pcds[source_id], pcds[target_id], source_id, target_id, odometry, pose_graph)
                                     for target_id in range(source_id + 1, n_pcds))
    print("end of full_registration");
    return pose_graph


if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds_down = load_point_clouds(voxel_size)
    pcds = pcds_down
    o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

    print("Transform points and display")
    pcd_combined_registered = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined_registered += pcds_down[point_id]
    o3d.io.write_point_cloud("pcd_dataset/pointcloud_registered1.ply", pcd_combined_registered);
    o3d.visualization.draw_geometries(pcds_down)

    print("Make a combined point cloud")
    #pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("pcd_dataset/pointcloud1.ply", pcd_combined_down);
    o3d.visualization.draw_geometries([pcd_combined_down])