"""
generate mesh from input img and pose
"""


from torch.utils.data import DataLoader
import torch

# from inout.logger import get_logger
from models.NOCS import ModelNOCS
from models.Baseline import ModelIFNOCS
from models.LBS import ModelLBSNOCS
from models.SegLBS import ModelSegLBS
from models.PMLBS import PMLBS
from models.LNR import ModelLNRNET
from models.MLNR import ModelMLNRNet
from models.MVMPLNR import ModelMVMPLNRNet

from config import get_cfg
from utils.DataUtils import *
import argparse


# preparer configuration
cfg = get_cfg()

task = cfg.TASK

if task == "lbs":
    Model = ModelLBSNOCS(cfg)
if task == "lbs_seg":
    Model = ModelSegLBS(cfg)
if task == "occupancy":    
    Model = ModelIFNOCS(cfg)

if task == "sapien_lbs":
    Model = PMLBS(cfg)
if task == "lnrnet":    
    Model = ModelLNRNET(cfg)
if task == "mlnrnet":    
    Model = ModelMLNRNet(cfg)
if task == "mvmp":    
    Model = ModelMVMPLNRNet(cfg)

device = torch.device(cfg.GPU)



def load_img(path):
    img = imread_rgb_torch(item_path, Size=self.img_size).type(torch.FloatTensor)
    print(img)


if __name__ == "__main__":
    
    """
    If  use multi-view, should give a directory, contains different views of a same instance
    """

    single_views = ['lnrnet', 'occupancy']
    multi_views = ['mlnrnet', 'mvmp']
    inputs = cfg.GEN_INPUT
    if os.path.isfile(inputs):
        assert task in single_views
        data = load_img(path)
    else:
        assert task in multi_views
        items = os.listdir(inputs)
        multi_instance = any([os.path.isdir(item) for item in items])
        if multi_instance:
            print("TODO")
            exit()
        else:
            imgs = [load_img(i) for i in items]
            view_num = len(imgs)
            data = {}
            data['colr00'] = imgs
            data['translation'] = [torch.Tensor((-0.5, -0.5, -0.5)), ] * view_num
            data['scale'] = [torch.Tensor([1]), ] * view_num

            grid_coords = Model.net.grid_coords
            grid_points_split = torch.split(grid_coords, 100000, dim=1)