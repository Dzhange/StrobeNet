import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.DataUtils import *
from models.loss import *
from time import time



def get_pair_crr(view1, view2, tar_mask1, tar_mask2, crr_idx, crr_mask):
    """
    find the correspondinig points between view1 and view2
    tar_mask: ground truth mask for each view

    """
    base_pn_pc = view1[:, tar_mask1.squeeze() > 0].transpose(1, 0)
    query_pn_pc = view2[:, tar_mask2.squeeze() > 0].transpose(1, 0)
    
    paired_pc1 = base_pn_pc[pair_idx]
    paired_pc2 = query_pn_pc[pair_idx].unsqueeze(0)

    masked_p1 = paired_pc1[0, mask.to(dtype=bool), :].cpu().detach().numpy()
    masked_p2 = paired_pc2[0, mask.to(dtype=bool), :].cpu().detach().numpy()

    diff = (masked_p1 - masked_p2) ** 2
    return maked_p1, masked_p2, diff

def get_crr(output_list, tar_mask, crr):    
    for b_id in range(batch_size):            
        for base_view_id in range(len(crr['crr-idx-mtx'])):
            for query_view_id in range(len(crr['crr-idx-mtx'][base_view_id])):