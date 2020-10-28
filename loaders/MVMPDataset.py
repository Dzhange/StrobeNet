"""
Multi-view Multi-shape dataset
In this dataset we reconstrcut the shape with each pose
"""

import os, sys, argparse, zipfile, glob, random, pickle, math
import numpy as np
import torch
from itertools import groupby
from sklearn.neighbors import NearestNeighbors

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
from loaders.MVSAPIENDataset import MVSAPIENDataset
from utils.DataUtils import *

class MVMPDataset(MVSAPIENDataset):

    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, idx):        
        data_list = []
        batch = {}
        frame_base_path = self.frame_ids[idx]

        for view in range(self.view_num):
            data = self.get_sv_data(frame_base_path, view)
            data_list.append(data)
        for k in data_list[0].keys():
            batch[k] = [item[k] for item in data_list]

        cano_occ = self.load_occupancies(frame_base_path)
        batch['cano_grid_coords'] = cano_occ['grid_coords']
        batch['cano_occupancies'] = cano_occ['occupancies']
        if self.config.TRANSFORM:
            batch['cano_translation'] = cano_occ['translation']
            batch['cano_scale'] = cano_occ['scale']

        # TODO: also include pair-wise consistency data
        if self.config.CONSISTENCY != 0:
            crr = self.get_crr(batch)
            data.update(crr)

        return batch

if __name__ == '__main__':
    import argparse
    from config import get_cfg
    # preparer configuration
    cfg = get_cfg()
    # f_str = ["color00", "nox00", "pnnocs00", "linkseg"]
    f_str = None
    Data = MVMPDataset(cfg, train=False)
    DataLoader = torch.utils.data.DataLoader(Data, batch_size=1, shuffle=True, num_workers=4)
    for i, Data in enumerate(DataLoader, 0):  # Get each batch
        # print(Data['color00'][0].to(device="cuda:0"))
        # print("\r {}".format(i))
        pass
        