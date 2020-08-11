import torch.nn as nn
import torch 
import torch.optim as optim

import os, sys, argparse, math, glob, gc, traceback
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../models'))
from tk3dv.ptTools import ptUtils
from tk3dv.ptTools import ptNets

# from SegNet import SegNet
# from IFNet import SVR

sys.path.append(os.path.join(FileDirPath, '../nrnocs/tools'))
import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
from pc2voxel import * 

point_cloud = torch.Tensor(np.random.normal(0,1,(2,3,3000)))
# point_cloud = torch.Tensor(np.random.normal(0,1,(2,3000,3)))
voxelize(point_cloud)


# bb_min = -0.5
# bb_max = 0.5
# resolution = 128
# grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
# grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()
# a = bb_max + bb_min
# b = bb_max - bb_min
# grid_coords = 2 * grid_points - a
# grid_coords = grid_coords / b
# grid_coords = torch.from_numpy(grid_coords)
# grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3))


# print(grid_coords)
# print(grid_coords.shape)