import torch
from models.loss import JointLoss


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
    loss = JointLoss()
    l = loss(output, target)
    print(l)

test()









