# Copyright (c) 2020 by BionicLab. All Rights Reserved.
# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
@File: Realsense
@Author: Haokun Wang
@Date: 2020/4/3 15:47
@Description:
The pyrealsense2 API can be found in https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html.
"""
import pyrealsense2 as rs
import yaml
import numpy as np
import time

class RealSense(object):
    # initialization from camera configuration file
    def __init__(self, camera_configuration_file_path):
        self._cfg = yaml.load(open(camera_configuration_file_path, 'r'), Loader=yaml.FullLoader)
        self.width = self._cfg["width"]
        self.height = self._cfg["height"]
        self.fps = self._cfg["fps"]
        self.serial_number = self._cfg["serial_number"]

        self.points = rs.points()
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number != '' and self.serial_number is not None:
            print('Config for realsense %s' % self.serial_number)

        ''' Configure depth and color streams '''
        # the resolution of color image can be higher than depth image
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Start streaming
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.distortion_coeffs = intrinsics.coeffs
        self.intrinsics = {'cx': intrinsics.ppx, 'cy': intrinsics.ppy, 'fx': intrinsics.fx, 'fy': intrinsics.fy}

        self.scale = self.get_depth_scale()

        print("------ Realsense start info ------ ")
        print("Frame size: " + str(self.width) + "*" + str(self.height))
        print("FPS: " + str(self.fps))
        print("Color frame format: ", rs.format.bgr8)
        print("Depth frame format: ", rs.format.z16)
        print("Depth scale: " + str(self.scale))
        print("Intrinsics: " + str(self.intrinsics))
        print("---------------------------------- ")

    """
    @Description: get images from a camera
    @parameters[in]: None
    @return:
        color_image: color image
        depth_image: depth image, unit:m
        point_cloud: point cloud ,unit:m
        infrared_L, infrared_R: left infrared image and right infrared image
    """

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        aligned_irL_frame = aligned_frames.get_infrared_frame(1)
        aligned_irR_frame = aligned_frames.get_infrared_frame(2)

        # Convert images to numpy arrays
        depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = depth_image_raw * self.scale
        color_image = np.asanyarray(aligned_color_frame.get_data())
        infrared_L = np.asanyarray(aligned_irL_frame.get_data())
        infrared_R = np.asanyarray(aligned_irR_frame.get_data())

        pc = rs.pointcloud()
        point_cloud = pc.calculate(aligned_depth_frame)
        # point_cloud = None

        result = {'color_image': color_image, 'depth_image_raw': depth_image_raw, 'depth_image': depth_image, 'point_cloud': point_cloud, 'infrared': [infrared_L, infrared_R]}

        return result

    """
    @Description: get intrinsics attributes of a camera
    @parameters[in]: None
    @return:
        intrinsics.fx: focal length of the image in width(columns)
        intrinsics.fy: focal length of the image in height(rows)
        intrinsics.ppx: the pixel coordinates of the principal point (center of projection) in width
        intrinsics.ppy: the pixel coordinates of the principal point (center of projection) in height
    """

    def get_intrinsics(self):
        return self.intrinsics["fx"], self.intrinsics["fy"], self.intrinsics["cx"], self.intrinsics["cy"], self.distortion_coeffs

    def get_intrinsics_matrix(self):
        return np.array(
            [self.intrinsics["fx"], 0, self.intrinsics["cx"], 0, self.intrinsics["fy"], self.intrinsics["cy"], 0, 0,
             1]).reshape(3, 3)

    def get_distortion_coeffs(self):
        return np.array(self.distortion_coeffs)

    def get_device(self):
        return self.profile.get_device()

    def get_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def get_serial_number(self):
        device = self.get_device()
        return str(device.get_info(rs.camera_info.serial_number))

    def stop(self):
        self.pipeline.stop()
        print("RealSense D435 stops now.")

import os
import sys

if __name__ == '__main__':
    _root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root_path)
    os.chdir(_root_path)
    print('work_dir: ', _root_path)

    camera = RealSense(
        './Config/rs_d435.yaml')
    frame = camera.get_frame()
    depth = frame.depth_image
    pc = frame.point_cloud
    print(depth[0])
    print(camera.get_intrinsics())
    print(camera.get_device())
    print(camera.get_depth_scale())
    print(camera.get_serial_number())
    print(pc[0])
    # show point cloud
    show_pc = False
    if show_pc:
        vtx = np.asarray(pc[0].get_vertices())
        vtx = np.array(vtx.tolist())
        cc = frame.color_image[0]
        # BGR to RGB
        cc = cc[:, :, ::-1]
        cc = cc.reshape(-1, 3)

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(cc.astype(np.float32) / 255)

        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox.min_bound = [-1, -1, 0.3]
        bbox.max_bound = [1, 0.6, 1.2]
        pcd = pcd.crop(bbox)
        o3d.visualization.draw_geometries([pcd], window_name='model', point_show_normal=True)
