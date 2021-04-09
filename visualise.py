import cv2
import numpy as np
import torch
import open3d as o3d
import time
# conda install -c open3d-admin open3d

def writeToPly(point_cloud, filename):
    # point_cloud is HxWx3
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0] * point_cloud.shape[1]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for c in range(point_cloud.shape[1]):
            for r in range(point_cloud.shape[0]):
                f.write(f"{point_cloud[r, c, 0]} {point_cloud[r, c, 1]} {point_cloud[r, c, 2]}\n")

class Visualiser:
    def __init__(self, Q):
        self.Q = Q
        self.initialized_flag = False

        self.pcd = o3d.geometry.PointCloud()

    def init(self, disparity, img):
        # for initialising windows
        print("Initialising Visualiser...")

        # arrays for use in generating the pointcloud
        self.xyz = np.zeros((disparity.shape[0] * disparity.shape[1], 3))
        self.rgb = np.zeros((disparity.shape[0] * disparity.shape[1], 3))

        cv2.namedWindow("Disparity", cv2.WINDOW_AUTOSIZE)
        self.update_pcd(disparity, img)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

        # seems broken
        # vis_viewctl = self.vis.get_view_control()
        # viewpoint = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
        # vis_viewctl.convert_from_pinhole_camera_parameters(viewpoint, allow_arbitrary=True)


    def update(self, disparity, img):
        if not self.initialized_flag:
            self.init(disparity, img)
            self.initialized_flag = True
        print("Updating Visualiser...")
        dispNormalised = disparity / np.max(disparity)
        # print(f"disp - shape {disp.shape}, max {np.max(disp)}, min {np.min(disp)}")
        cv2.imshow("Disparity", dispNormalised)

        # update pointcloud
        self.update_pcd(disparity, img)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        # param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)


    def update_pcd(self, disparity, img):
        # disparity should be (HxW)
        point_cloud = cv2.reprojectImageTo3D(disparity, self.Q)  # outputs HxWx3
        # writeToPly(point_cloud, "test.ply")
        # pointcloud
        # normalise img values to [0, 1]
        img = img.astype(np.float32) / 255.
        self.xyz = np.reshape(point_cloud, self.xyz.shape, 'C')
        self.rgb = np.reshape(img, self.rgb.shape, 'C')
        self.pcd.points = o3d.utility.Vector3dVector(self.xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(self.rgb)


    def release(self):
        self.vis.destroy_window()
