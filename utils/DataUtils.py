"""
This file contains some useful utils
for data loading and processing
"""
import os
import glob
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from models.loss import L2MaskLoss
import re
    
def find_frame_num(path):
    return re.findall(r'%s(\d+)' % "frame_",path)[0]

################################ DATA RELATED UTILS ####################################
def torch2np(ImageTorch):
    # OpenCV does [height, width, channels]
    # PyTorch stores images as [channels, height, width]
    if ImageTorch.size()[0] == 3:
            return np.transpose(ImageTorch.numpy(), (1, 2, 0))
    return ImageTorch.numpy()

def np2torch(ImageNP):
    # PyTorch stores images as [channels, height, width]
    # OpenCV does [height, width, channels]
    if ImageNP.shape[-1] == 3:
            return torch.from_numpy(np.transpose(ImageNP, (2, 0, 1)))
    return torch.from_numpy(ImageNP)

def imread_rgb_torch(Path, Size=None, interp=cv2.INTER_NEAREST): # Use only for loading RGB images
    ImageCV = cv2.imread(Path, -1)
    # Discard 4th channel since we are loading as RGB
    if ImageCV.shape[-1] != 3:
        ImageCV = ImageCV[:, :, :3]

    ImageCV = cv2.cvtColor(ImageCV, cv2.COLOR_BGR2RGB)
    if Size is not None:
        # Check if the aspect ratios are the same
        OrigSize = ImageCV.shape[:-1]
        OrigAspectRatio = OrigSize[1] / OrigSize[0] # W / H
        ReqAspectRatio = Size[0] / Size[1] # W / H # CAUTION: Be aware of flipped indices
        # print(OrigAspectRatio)
        # print(ReqAspectRatio)
        if math.fabs(OrigAspectRatio-ReqAspectRatio) > 0.01:
            # Different aspect ratio detected. So we will be fitting the smallest of the two images into the larger one while centering it
            # After centering, we will crop and finally resize it to the request dimensions
            if ReqAspectRatio < OrigAspectRatio:
                NewSize = [OrigSize[0], int(OrigSize[0] * ReqAspectRatio)] # Keep height
                Center = int(OrigSize[1] / 2) - 1
                HalfSize = int(NewSize[1] / 2)
                ImageCV = ImageCV[:, Center-HalfSize:Center+HalfSize, :]
            else:
                NewSize = [int(OrigSize[1] / ReqAspectRatio), OrigSize[1]] # Keep width
                Center = int(OrigSize[0] / 2) - 1
                HalfSize = int(NewSize[0] / 2)
                ImageCV = ImageCV[Center-HalfSize:Center+HalfSize, :, :]
        ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=interp)
    Image = np2torch(ImageCV) # Range: 0-255

    return Image

def imread_gray_torch(Path, Size=None, interp=cv2.INTER_NEAREST): # Use only for loading RGB images
    ImageCV = cv2.imread(Path, -1)
    # Discard 4th channel since we are loading as RGB
    if ImageCV.shape[-1] != 3:
        ImageCV = ImageCV[:, :, :3]

    ImageCV = cv2.cvtColor(ImageCV, cv2.COLOR_BGR2GRAY)
    if Size is not None:
        # Check if the aspect ratios are the same
        OrigSize = ImageCV.shape[:]        
        OrigAspectRatio = OrigSize[1] / OrigSize[0] # W / H
        ReqAspectRatio = Size[0] / Size[1] # W / H # CAUTION: Be aware of flipped indices
        # print(OrigAspectRatio)
        # print(ReqAspectRatio)
        if math.fabs(OrigAspectRatio-ReqAspectRatio) > 0.01:
            # Different aspect ratio detected. So we will be fitting the smallest of the two images into the larger one while centering it
            # After centering, we will crop and finally resize it to the request dimensions
            if ReqAspectRatio < OrigAspectRatio:
                NewSize = [OrigSize[0], int(OrigSize[0] * ReqAspectRatio)] # Keep height
                Center = int(OrigSize[1] / 2) - 1
                HalfSize = int(NewSize[1] / 2)
                ImageCV = ImageCV[:, Center-HalfSize:Center+HalfSize, :]
            else:
                NewSize = [int(OrigSize[1] / ReqAspectRatio), OrigSize[1]] # Keep width
                Center = int(OrigSize[0] / 2) - 1
                HalfSize = int(NewSize[0] / 2)
                ImageCV = ImageCV[Center-HalfSize:Center+HalfSize, :, :]
        ImageCV = cv2.resize(ImageCV, dsize=Size, interpolation=interp)
    
    Image = torch.from_numpy(np.transpose(ImageCV, (0, 1)))
    return Image

def saveData(Items, OutPath='.'):
    for Ctr, I in enumerate(Items, 0):
        if len(I.shape) == 3:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(OutPath, 'item' + str(Ctr).zfill(4) + '.png'), I)

def createMask(NOCSMap):
    LocNOCS = NOCSMap.type(torch.FloatTensor)

    Norm = torch.norm(LocNOCS, dim=0)
    ZeroT = torch.zeros((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device)
    OneT = torch.ones((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device) * 255 # Range: 0, 255
    Mask = torch.where(Norm >= 441.6729, ZeroT, OneT)  # 441.6729 == sqrt(3 * 255^2)
    return Mask.type(torch.FloatTensor)

def applyMask(NOX, Thresh):
    # Input (only torch): 4-channels where first 3 are NOCS, last is mask
    # Output (numpy): 3-channels where the mask is used to mask out the NOCS
    if NOX.size()[0] != 4:
        raise RuntimeError('[ ERR ]: Input needs to be a 4 channel image.')

    NOCS = NOX[:3, :, :]
    Mask = NOX[3, :, :]

    MaskProb = torch2np(torch.squeeze(F.sigmoid(Mask)))
    Masked = np.uint8((MaskProb > Thresh) * 255)
    MaskedNOCS = torch2np(torch.squeeze(NOCS))
    MaskedNOCS[MaskProb <= Thresh] = 255

    return MaskedNOCS, Masked

def convertData(RGB, Targets, isMaskNOX=False):
    TargetsTup = Targets
    Color00 = torch2np(RGB[0:3].squeeze()).squeeze() * 255

    OutTupRGB = ()
    OutTupMask = ()
    # Convert range to 0-255
    for T in TargetsTup:
        T = T.squeeze() * 255
        nChannels = T.size(0)
        if nChannels != 4 and nChannels != 8:
            raise RuntimeError('Only supports 4/8-channel input. Passed {}'.format(nChannels))

        if nChannels == 4:
            if isMaskNOX:
                MaskedT, _ = applyMask(torch.squeeze(T[0:4]), Thresh=L2MaskLoss.Thresh)
                OutTupRGB = OutTupRGB + (MaskedT,)
            else:
                OutTupRGB = OutTupRGB + (torch2np(T[0:3]).squeeze(),)
            OutTupMask = OutTupMask + (torch2np(T[3]).squeeze(),)
        else:
            if isMaskNOX:
                MaskedT0, _ = applyMask(torch.squeeze(T[0:4]), Thresh=L2MaskLoss.Thresh)
                MaskedT1, _ = applyMask(torch.squeeze(T[4:8]), Thresh=L2MaskLoss.Thresh)
                OutTupRGB = OutTupRGB + (MaskedT0, MaskedT1)
            else:
                OutTupRGB = OutTupRGB + (torch2np(T[0:3]).squeeze(), torch2np(T[4:7]).squeeze())
            OutTupMask = OutTupMask + (torch2np(T[3]).squeeze(), torch2np(T[7]).squeeze())

    return Color00, OutTupRGB, OutTupMask

def sendToDevice(TupleOrTensor, Device):
    TupleOrTensorTD = TupleOrTensor
    if isinstance(TupleOrTensorTD, tuple) == False and isinstance(TupleOrTensorTD, list) == False:
        TupleOrTensorTD = TupleOrTensor.to(Device)
    else:
        for Ctr in range(len(TupleOrTensor)):
            if isinstance(TupleOrTensor[Ctr], torch.Tensor):
                TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr].to(Device)
            else:
                TupleOrTensorTD[Ctr] = TupleOrTensor[Ctr]

    return TupleOrTensorTD

def normalizeInput(Image, format='imagenet'):
    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be atleast 224.
    # The images have to be loaded in to a range of [0, 1] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

    ImageN = Image # Assuming that input is in 0-1 range already
    if 'imagenet' in format:
        # Apply ImageNet batch normalization for input
        # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560
        # Assuming torch image 3 x W x H
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ImageN[0] = (ImageN[0] - 0.485 ) / 0.229
        ImageN[1] = (ImageN[1] - 0.456 ) / 0.224
        ImageN[2] = (ImageN[2] - 0.406 ) / 0.225
    else:
        print('[ WARN ]: Input normalization implemented only for ImageNet.')

    return ImageN

def expandTilde(Path):
    if '~' == Path[0]:
        return os.path.expanduser(Path)

    return Path


################################ TIME RELATED UTILS ####################################

def getCurrentEpochTime():
    return int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() * 1e6)

def getZuluTimeString(StringFormat = '%Y-%m-%dT%H-%M-%S'):
    return datetime.utcnow().strftime(StringFormat)

def getLocalTimeString(StringFormat = '%Y-%m-%dT%H-%M-%S'):
    return datetime.now().strftime(StringFormat)

def getTimeString(TimeString='humanlocal'):
    TS = TimeString.lower()
    OTS = 'UNKNOWN'

    if 'epoch' in TS:
        OTS = str(getCurrentEpochTime())
    else:
        if 'zulu' in TS:
            OTS = getZuluTimeString(StringFormat='%Y-%m-%dT%H-%M-%SZ')
        elif 'local' in TS:
            OTS = getLocalTimeString(StringFormat='%Y-%m-%dT%H-%M-%S')
        elif 'eot' in TS: # End of time
            OTS = '9999-12-31T23-59-59'

    if 'human' in TS:
        OTS += '_' + str(getCurrentEpochTime())

    return OTS

def dhms(td):
    d, h, m = td.days, td.seconds//3600, (td.seconds//60)%60
    s = td.seconds - ( (h*3600) + (m*60) ) # td.seconds are the seconds remaining after days have been removed
    return d, h, m, s

def getTimeDur(seconds):
    Duration = timedelta(seconds=seconds)
    OutStr = ''
    d, h, m, s = dhms(Duration)
    if d > 0:
        OutStr = OutStr + str(d)+ ' d '
    if h > 0:
        OutStr = OutStr + str(h) + ' h '
    if m > 0:
        OutStr = OutStr + str(m) + ' m '
    OutStr = OutStr + str(s) + ' s'

    return OutStr

############################### PYTORCH UTILS ######################################
def loadPyTorchCheckpoint(InPath, map_location='cpu'):
    return torch.load(expandTilde(InPath), map_location=map_location)

def loadLatestPyTorchCheckpoint(InDir, CheckpointName='', map_location='cpu'):
    AllCheckpoints = glob.glob(os.path.join(expandTilde(InDir), CheckpointName + '*.tar'))
    if len(AllCheckpoints) <= 0:
        raise RuntimeError('No checkpoints stored in ' + InDir)
    AllCheckpoints.sort() # By name

    print('[ INFO ]: Loading checkpoint {}'.format(AllCheckpoints[-1]))
    return loadPyTorchCheckpoint(AllCheckpoints[-1], map_location)

def savePyTorchCheckpoint(CheckpointDict, OutDir, TimeString='humanlocal'):
    # CheckpointDict should have a model name, otherwise, will store as UNKNOWN
    Name = 'UNKNOWN'
    if 'Name' in CheckpointDict:
        Name = CheckpointDict['Name']

    OTS = getTimeString(TimeString)
    OutFilePath = os.path.join(expandTilde(OutDir), Name + '_' + OTS + '.tar')
    torch.save(CheckpointDict, OutFilePath)
    return OutFilePath  

def saveLossesCurve(*args, **kwargs):
    plt.clf()
    ylim = 0
    for arg in args:
        if len(arg) <= 0:
            continue
        plt.plot(arg)
        ylim = ylim + np.median(np.asarray(arg))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'legend' in kwargs:
        if len(kwargs['legend']) > 0:
            plt.legend(kwargs['legend'])
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    if ylim > 0:
        plt.ylim([0.0, ylim])
    if 'out_path' in kwargs:
        plt.savefig(kwargs['out_path'])
    else:
        print('[ WARN ]: No output path (out_path) specified. ptUtils.saveLossesCurve()')