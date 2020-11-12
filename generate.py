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

Model.net.to(device=device)

def load_img(item_path):
    print(item_path)
    img = imread_rgb_torch(item_path, Size=cfg.IMAGE_SIZE).type(torch.FloatTensor)
    # print(img.shape)
    img /= 255.0

    return img.unsqueeze(0)


def save_img(output_dir, net_input, output, i=0, view_id=0):    
    input_img, pred_out_tuple_rgb, pred_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
    cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_color00.png').format(str(i).zfill(3), view_id), cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    out_target_str = ['nocs']

    for target_str, pred, pred_mask in zip(out_target_str, pred_out_tuple_rgb, pred_out_tuple_mask):        
        cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_{}_01pred.png').format(str(i).zfill(3), view_id, target_str),
                    cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))        
        cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_{}_03predmask.png').format(str(i).zfill(3), view_id, target_str),
                    pred_mask)


if __name__ == "__main__":
    
    """
    If  use multi-view, should give a directory, contains different views of a same instance
    """

    single_views = ['lnrnet', 'occupancy']
    multi_views = ['mlnrnet', 'mvmp']
    inputs = cfg.GEN_INPUT

    # single image input
    if os.path.isfile(inputs):
        assert task in single_views
        data = load_img(path)
    # multi view
    else:
        assert task in multi_views
        items = os.listdir(inputs)
        multi_instance = any([os.path.isdir(item) for item in items])
        if multi_instance:
            print("TODO")
            exit()
        else:
            imgs = [load_img(os.path.join( inputs, i)).to(device=device) for i in items]
            view_num = len(imgs)
            net_input = {}
            net_input['color00'] = imgs
            net_input['translation'] = [torch.Tensor((-0.5, -0.5, -0.5)).unsqueeze(0).to(device=device), ] * view_num
            net_input['scale'] = [torch.Tensor([1]).unsqueeze(0).to(device=device), ] * view_num            

            # grid_coords = Model.net.grid_coords
            grid_coords = Model.net.init_grids(Model.net.resolution)
            grid_points_split = torch.split(grid_coords, 100000, dim=1)
    
        logits_list = []        

        for points in grid_points_split:
            with torch.no_grad():
                # net_input, target = self.preprocess(data, device)
                # print(points.shape)
                net_input['grid_coords'] = [points.to(device), ] * cfg.VIEW_NUM
                net_input['cano_grid_coords'] = points.to(device)
                
                output = Model.net(net_input)
                
                save_img("../debug", net_input['color00'][0], output[0][0][:, 0:4, :, :])
                # exit()
                print(output[1].sum())
                logits_list.append(output[1].squeeze(0).detach().cpu())

        # generate predicted mesh from occupancy and save
        logits = torch.cat(logits_list, dim=0).numpy()
        mesh = Model.mesh_from_logits(logits, Model.net.resolution)
        # export_pred_path = os.path.join(Model.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
        export_pred_path = "/workspace/test.obj"
        mesh.export(export_pred_path)