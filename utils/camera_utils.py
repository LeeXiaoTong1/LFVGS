#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from scene.cameras import Camera
import numpy as np
import cv2
from tqdm import tqdm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from depth_utils import estimate_depth
from utils.depth_normal import depth_to_normal

WARNED = False
def savePILImageAsPNG(pil_image, save_path):
    pil_image.save(save_path, format='PNG')
def loadCam(args, id, cam_info, resolution_scale, depth_model):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    # 将 PIL 图像转换为 PNG 文件并读取回来
    temp_dir = "/tmp"
    temp_image_path = os.path.join(temp_dir, f"{cam_info.image_name}.png")  # 使用原始名称
    savePILImageAsPNG(cam_info.image, temp_image_path)
    raw_img = cv2.imread(temp_image_path)
    if raw_img is None:
        raise ValueError(f"Could not load image from temporary file: {temp_image_path}")
    # 打印 raw_img 的形状
    # print(f"Raw image shape: {raw_img.shape}")
    depth = estimate_depth(raw_img, depth_model)
    
    # 动态计算焦距
    focal_x = fov2focal(cam_info.FovX, orig_w)
    focal_y = fov2focal(cam_info.FovY, orig_h)
    
    # 深度转法线
    normal = depth_to_normal(depth, cam_info.FovX, cam_info.FovY, orig_w, orig_h)


    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    mask = None if cam_info.mask is None else cv2.resize(cam_info.mask, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  image=gt_image, gt_alpha_mask=loaded_mask,
                  uid=id, data_device=args.data_device, image_name=cam_info.image_name,
                  depth_image=depth, normal_map=normal, mask=mask, bounds=cam_info.bounds)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, depth_model):
    camera_list = []

    for id, c in tqdm((enumerate(cam_infos))):
        camera_list.append(loadCam(args, id, c, resolution_scale, depth_model))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
