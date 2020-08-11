"""
Generate some validation results for visualization
"""
import os
import traceback
import torch
from utils.DataUtils import *
from loaders.HandDataset import *

class Validater:

    def __init__(self, config, model, val_dataloader, objective, device):
        self.config = config
        self.model = model
        self.objective = objective
        self.device = device
        self.val_dataloader = val_dataloader                
        
        self.model.net.to(device)
        self.output_dir = os.path.join(self.model.expt_dir_path, "ValResults")
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)

    def validate(self):
        self.model.setup_checkpoint(self.device)
        self.model.net.eval()

        ValLosses = []
        
        max_num = 30
        epoch_losses = []
        for i, data in enumerate(self.val_dataloader, 0):  # Get each batch        
            if i >= max_num:
                break
            
            net_input, target = self.model.preprocess(data, self.device)
            output = self.model.net(net_input)
            loss = self.objective(output, target)
            epoch_losses.append(loss.item())

            print("validating on the {}th data, loss is {}".format(i,loss))
            print("average validation loss is ",np.mean(np.asarray(epoch_losses)))            

            InputIm, GTOutTupRGB, GTOutTupMask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(target, 'cpu'))
            _, PredOutTupRGB, PredOutTupMask = convertData(sendToDevice(net_input, 'cpu'), sendToDevice(output.detach(), 'cpu'), isMaskNOX=True)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_color00.png').format(str(i).zfill(3)),  cv2.cvtColor(InputIm, cv2.COLOR_BGR2RGB))

            OutTargetStr = [self.config.TARGETS]
            if 'color00' in OutTargetStr:
                OutTargetStr.remove('color00')
            if 'camera' in OutTargetStr:
                OutTargetStr.remove('camera')

            for Targetstr, GT, Pred, GTMask, PredMask in zip(OutTargetStr, GTOutTupRGB, PredOutTupRGB, GTOutTupMask, PredOutTupMask):
                cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_00gt.png').format(str(i).zfill(3), Targetstr),
                            cv2.cvtColor(GT, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_01pred.png').format(str(i).zfill(3), Targetstr),
                            cv2.cvtColor(Pred, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_02gtmask.png').format(str(i).zfill(3), Targetstr),
                            GTMask)
                cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_{}_03predmask.png').format(str(i).zfill(3), Targetstr),
                            PredMask)
        
        print("average validation loss is ", np.mean(np.asarray(epoch_losses)))
            