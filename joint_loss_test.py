import numpy as np
import torch
from models.loss import LBSLoss


def test():        
    batch_size = 2
    bone_num = 16
    out_num = 3 + 1 + 3 + bone_num*3 + bone_num
    tar_num = 3 + 1 + bone_num*3
    s1, s2 = 240, 320

    output = torch.zeros(batch_size, out_num, s1, s2)
    target_map = torch.zeros(batch_size, tar_num, s1, s2)
    target_joint = torch.zeros(batch_size, bone_num, 3)
    target = {
        'map':target_map,
        'pose':target_joint
        }
    loss = LBSLoss()
    l = loss(output, target)
    print(l)


def pose2maps():

    pose = np.loadtxt("/workspace/Data/8_7_cut_BW/train/0000/frame_00000000_hpose_rel.txt")
    pose = torch.Tensor(pose)
    
    joint_map = ()
    for i in range(pose.shape[0]):
        for j in range(6):
            cur_map = torch.Tensor(*(320,240)).unsqueeze(0)
            print(cur_map.shape)
            joint_map = joint_map + (cur_map.fill_(pose[i,j]),)
    maps = torch.cat(joint_map, 0)
    print(maps.shape)
pose2maps()
# test()











