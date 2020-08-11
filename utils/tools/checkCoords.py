import implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os,re
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import random
import traceback
import cv2


p = "/ZG/Data/frame_00000000_view_00_test/frame_00001000_view_00_nox00.png"


nocs = cv2.imread(p)
nocs = cv2.cvtColor(nocs,cv2.COLOR_BGR2RGB)

valid_idx = np.where(np.all(nocs != [255, 255, 255], axis=-1)) # Only white BG
num_valid = valid_idx[0].shape[0]

randomIdx = np.random.choice(num_valid,sample_num // len(nocs_path),replace=False)
# for current use we choose uniform sample
sampled_idx = (valid_idx[0][randomIdx], valid_idx[1][randomIdx])
sample_points = nocs[sampled_idx[0], sampled_idx[1]] / 255


mp =  '/ZG/Data/frame_00000000_view_00_test/frame_00000000_isosurf_scaled.off'
mesh = trimesh.load(off_path)
point_cloud = mesh.sample(args.num_points)