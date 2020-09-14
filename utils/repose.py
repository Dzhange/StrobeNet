from pathlib import Path
import os
import os
import numpy as np
import cv2
import torch
from tk3dv.ptTools import ptUtils
from lbs import *
import trimesh


def write_off(path, pc):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'a') as f1:
        for i in range(pc.shape[0]):
            p = pc[i]
            if np.linalg.norm(p) < 500:
                f1.write("{} {} {}\n".format(p[0],p[1],p[2]))
    f1.close()


use_mesh = False
# use_mesh = True
bone_num = 16
base_bone_path = "D:/GuibasLab/Data/pose/0000/frame_00000000_view_00_BoneWeight00bone_"
NOCS_Path =      "D:/GuibasLab/Data/pose/0000/frame_00000000_view_00_nox00.png"
PNNOCS_Path =    "D:/GuibasLab/Data/pose/0000/frame_00000000_view_00_pnnocs00.png"
pose_path =     "D:/GuibasLab/Data/pose/0000/frame_00000000_hpose_nocs.txt"
canonical_path = "D:/GuibasLab/Code/repose_test/canonical.txt"

pose = np.loadtxt(pose_path)
joint_positions = torch.Tensor(pose[:,:3]).unsqueeze(0)
joint_axis_angle = torch.Tensor(pose[:,3:6]).unsqueeze(0)

canonical_pose = np.loadtxt(canonical_path)
can_pose = torch.Tensor(canonical_pose).unsqueeze(0)

# joint_axis_angle = torch.Tensor(pose[:,3:6])

parents = pose[:,6]
parents[np.where(parents==-1)] = 999
parents = torch.Tensor(parents).to(dtype=torch.int64)

# _, sort_index = torch.Tensor(parents).sort() #get dependency straight
# sort_index_list = sort_index.to(dtype=torch.int32).tolist()

# new_parents = [0]*16
# new_parents[0] = 999
# for i in range(1,16):
#     # new_parents[i] = sort_index_list.index(parents[i])
#     new_parents[i] = parents.index(sort_index_list[i])

# print(new_parents)
# parents = torch.Tensor(new_parents).to(dtype=torch.int64)

nocs = cv2.imread(NOCS_Path)
nocs = cv2.cvtColor(nocs,cv2.COLOR_BGR2RGB)
pnnocs = cv2.imread(PNNOCS_Path)
pnnocs = cv2.cvtColor(pnnocs,cv2.COLOR_BGR2RGB)

valid_idx = np.where(np.all(nocs != [255, 255, 255], axis=-1)) 

# Get point cloud for rest/deformed pose
nocs_pc = nocs[valid_idx[0], valid_idx[1]] / 255
pnnocs_pc = pnnocs[valid_idx[0], valid_idx[1]] / 255



nocs_pc = torch.Tensor(nocs_pc).unsqueeze(0)
# nocs_pc = torch.Tensor(pnnocs_pc).unsqueeze(0)

bone_weights = []
for i in range(bone_num):
    bm = cv2.imread(base_bone_path +  str(i) + ".png")
    bm = cv2.cvtColor(bm,cv2.COLOR_RGB2GRAY)
    bw = bm[valid_idx[0], valid_idx[1]] / 255
    bone_weights.append(torch.Tensor(bw))

bone_pixel_weights = torch.stack(bone_weights)
bone_pixel_weights = bone_pixel_weights.transpose(0,1) # Bone pixel weights now have shape NXJ

if not use_mesh:
    valid_pixels = bone_pixel_weights.sum(dim=1)>0.99
    bone_pixel_weights = bone_pixel_weights[valid_pixels, :]
    nocs_pc = nocs_pc[:, valid_pixels, :]

    origin_path = "./origin.xyz"
    write_off(origin_path, nocs_pc[0])

    write_off("./origin_joints.xyz", joint_positions[0])

    repose_angle = -joint_axis_angle

    T,T_inv = rotation_matrix(repose_angle, joint_positions, bone_pixel_weights, parents, dtype=nocs_pc.dtype)
    
    output_ = lbs_(nocs_pc.to(dtype=torch.float64), T.to(dtype=torch.float64), dtype=torch.float64)     

    reposed_path = './reposed.xyz'
    write_off(reposed_path, output_[0])
else:
    origin_mesh = "D:/GuibasLab/Data/pose/0000/frame_00000000_World_mesh.obj"
    mesh_bw = "D:/GuibasLab/Data/pose/0000/frame_00000000_mesh_BW.txt"
    mesh = trimesh.load(origin_mesh, process=False)
    
    mesh_points = mesh.vertices
    origin_path = "./origin_mesh.xyz"
    write_off(origin_path, mesh_points)
    nocs_pc = torch.Tensor(mesh_points).unsqueeze(0)
    bone_weights = torch.Tensor(
        np.loadtxt(mesh_bw)[:,0:16]
    )

    T,T_inv = rotation_matrix(joint_axis_angle, joint_positions, bone_weights, parents, dtype=nocs_pc.dtype)
    output_ = lbs_(nocs_pc.to(dtype=torch.float64), T.to(dtype=torch.float64), dtype=torch.float64)

    reposed_path = './reposed.xyz'
    write_off(reposed_path, output_[0])