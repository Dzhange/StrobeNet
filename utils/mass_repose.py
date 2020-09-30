'''
take the results of several task, put them together, 
do linear blend skinning to put it back into normal pose
1. nocs
2. skinning weights
3. 

'''
from pathlib import Path
import os
import os
import numpy as np
import cv2
import torch
from tk3dv.ptTools import ptUtils
from lbs import *
import trimesh


bone_num = 16
Base_dir = "F:/Pose916_addition/0000/"
frame = "00000000"
view = "00"

bone_num = 16
base_bone_path = Base_dir + "frame_" + frame + "_view_"+ view +"_BoneWeight00bone_"
NOCS_Path =      Base_dir + "frame_" + frame + "_view_"+ view +"_nox00.png"
PNNOCS_Path =    Base_dir + "frame_" + frame + "_view_"+ view +"_pnnocs00.png"
pose_path =      Base_dir + "frame_" + frame + "_hpose_nocs.txt"
canonical_path = Base_dir + "frame_" + frame + "_hpose_pnnocs.txt"

origin_path = "./origin.xyz"
reposed_path = './reposed.xyz'
normalized_path = './normalized.xyz'

pose = np.loadtxt(pose_path)
joint_positions = torch.Tensor(pose[:,:3]).unsqueeze(0)
joint_axis_angle = torch.Tensor(pose[:,3:6]).unsqueeze(0)

canonical_pose = np.loadtxt(canonical_path)
can_pose = torch.Tensor(canonical_pose).unsqueeze(0)

# joint_axis_angle = torch.Tensor(pose[:,3:6])

parents = pose[:,6]
parents[np.where(parents==-1)] = 999
parents = torch.Tensor(parents).to(dtype=torch.int64)

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

valid_pixels = bone_pixel_weights.sum(dim=1)>0.99
bone_pixel_weights = bone_pixel_weights[valid_pixels, :]
nocs_pc = nocs_pc[:, valid_pixels, :]

repose_angle = -joint_axis_angle
T,_, J_transformed = rotation_matrix(repose_angle, joint_positions, bone_pixel_weights, parents, dtype=nocs_pc.dtype)
zero_posed = lbs_(nocs_pc.to(dtype=torch.float64), T.to(dtype=torch.float64), dtype=torch.float64)     
can_T, _, J_transformed = rotation_matrix(can_pose, J_transformed, bone_pixel_weights, parents, dtype=nocs_pc.dtype)
normalized = lbs_(zero_posed, can_T.to(dtype=torch.float64), dtype=torch.float64)     


def write_off(path, pc):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'a') as f1:
        for i in range(pc.shape[0]):
            p = pc[i]
            if np.linalg.norm(p) < 500:
                f1.write("{} {} {}\n".format(p[0],p[1],p[2]))
    f1.close()

write_off(origin_path, nocs_pc[0])
write_off(reposed_path, zero_posed[0])
write_off(normalized_path, normalized[0])