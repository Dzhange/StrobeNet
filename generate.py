"""
generate mesh from input img and pose
"""


from torch.utils.data import DataLoader
import torch
from torchvision import transforms

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

from resizeimage import resizeimage
from PIL import Image, ImageColor, ImageOps




def resize_img(img_path, pix=150):
    img = Image.open(img_path)    
    img = ImageOps.expand(img, border=(pix,pix,pix,pix), fill=(255, 255, 255))##left,top,right,bottom
    img = resizeimage.resize_cover(img, [640, 480])
    img.save(img_path.replace('.png', '_padding.png').replace('.jpg', '_padding.png'))

def load_img(item_path):
    print("loading ", item_path)

    img = imread_rgb_torch(item_path, Size=cfg.IMAGE_SIZE).type(torch.FloatTensor)
    # print(img.shape)
    img /= 255.0

    mean = [0.46181917, 0.44638503, 0.40571916]
    std = [0.24907675, 0.24536176, 0.24932721]
    transform = transforms.Normalize(mean=mean, std=std)
    
    img = transform(img)

    return img.unsqueeze(0)

def save_img(output_dir, net_input, output, i=0, view_id=0):
    input_img, pred_out_tuple_rgb, pred_out_tuple_mask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
    cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_color00.png').format(str(i).zfill(3), view_id), cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    out_target_str = ['nocs']
    # print(output.shape)
    for target_str, pred, pred_mask in zip(out_target_str, pred_out_tuple_rgb, pred_out_tuple_mask):
        cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_{}_01pred.png').format(str(i).zfill(3), view_id, target_str),
                    cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
        
        # pred_mask = np.log(pred_mask / (1 - pred_mask))
        cv2.imwrite(os.path.join(output_dir, 'frame_{}_view_{}_{}_03predmask.png').format(str(i).zfill(3), view_id, target_str),
                    pred_mask)

def gen_NOCS_pc(output_dir, pred_nocs_map, mask, target_str="pred_nocs", i=0, view_id=0):
            
    mask = sendToDevice(mask.detach(), 'cpu')
    pred_nocs_map = sendToDevice(pred_nocs_map.detach(), 'cpu')
    
    # We use gt mask
    mask_prob = torch2np(mask.squeeze())
    pred_nocs_map = torch2np(torch.squeeze(pred_nocs_map))    

    pred_nocs = pred_nocs_map[mask_prob > 0.7]    
    
    pred_nocs_path = os.path.join(output_dir, 'frame_{}_view_{}_{}_01pred.xyz').format(str(i).zfill(3), view_id, target_str)
    write_off(pred_nocs_path, pred_nocs)





if __name__ == "__main__":
    
    """
    If  use multi-view, should give a directory, contains different views of a same instance
    """
    # preparer configuration
    cfg = get_cfg()
    cfg.defrost()
    cfg.TRAIN = False
    cfg.freeze()

    task = cfg.TASK

    single_views = ['lnrnet', 'occupancy']
    multi_views = ['mlnrnet', 'mvmp']
    inputs = cfg.GEN_INPUT

    if 0:
        resize_img(inputs, 200)
        exit()
    
    if task == "lnrnet":    
        Model = ModelLNRNET(cfg)
    elif task == "mlnrnet":    
        Model = ModelMLNRNet(cfg)
    elif task == "mvmp":    
        Model = ModelMVMPLNRNet(cfg)
    else:
        exit()

    device = torch.device(cfg.GPU)
    Model.net.to(device=device)
    Model.setup_checkpoint(device)
    # single image input
    if os.path.isfile(inputs):
        imgs = [load_img(inputs).to(device=device)]        
        view_num = 1
        multi_instance = False        
    # multi view
    else:
        assert task in multi_views
        items = os.listdir(inputs)
        imgs = [load_img(os.path.join( inputs, i)).to(device=device) for i in items]
        view_num = len(imgs)

        multi_instance = any([os.path.isdir(item) for item in items])
    # print(items)
    if multi_instance:
        print("TODO")
        exit()
    
    net_input = {}
    net_input['color00'] = imgs
    net_input['translation'] = [torch.Tensor((-0.5, -0.5, -0.5)).unsqueeze(0).to(device=device), ] * view_num
    net_input['scale'] = [torch.Tensor([1]).unsqueeze(0).to(device=device), ] * view_num            

    # grid_coords = Model.net.grid_coords
    grid_coords = Model.net.init_grids(Model.net.resolution)
    grid_points_split = torch.split(grid_coords, 100000, dim=1)

    logits_list = []        
        
    output = Model.net(net_input)
    # for points in grid_points_split:
    #     with torch.no_grad():
    #         # net_input, target = self.preprocess(data, device)
    #         # print(points.shape)
    #         net_input['grid_coords'] = [points.to(device), ] * cfg.VIEW_NUM
    #         net_input['cano_grid_coords'] = points.to(device)
            
    #         output = Model.net(net_input)
    outdir = "../debug"
    index = len(os.listdir(outdir))
    cur_out_dir = os.path.join(outdir, str(index).zfill(3))
    os.mkdir(cur_out_dir)
    save_img(cur_out_dir, net_input['color00'][0], output[0][0][:, 0:4, :, :])
    gen_NOCS_pc(cur_out_dir, output[0][0][:, 0:3, :, :], output[0][0][:, 3, :, :])
    #         # exit()
    #         print(output[1].sum())
    #         logits_list.append(output[1].squeeze(0).detach().cpu())

    # # generate predicted mesh from occupancy and save
    # logits = torch.cat(logits_list, dim=0).numpy()
    # mesh = Model.mesh_from_logits(logits, Model.net.resolution)
    # # export_pred_path = os.path.join(Model.output_dir, "frame_{}_recon.off".format(str(i).zfill(3)))
    # export_pred_path = "/workspace/test.obj"
    # mesh.export(export_pred_path)