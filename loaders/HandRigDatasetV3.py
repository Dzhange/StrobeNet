import torch
import torch.utils.data
import numpy as np
import json
import matplotlib.pyplot as plt
import os, sys, argparse, zipfile, glob, random, pickle, math
from itertools import groupby
import torch.nn.functional as F
import cv2

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from torch import nn


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

# This is the basic loader that loads all data without any model ID separation of camera viewpoint knowledge
class HandRigDatasetV3(torch.utils.data.Dataset):    
    
    @staticmethod
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

    @staticmethod
    def saveData(Items, OutPath='.'):
        for Ctr, I in enumerate(Items, 0):
            if len(I.shape) == 3:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(OutPath, 'item' + str(Ctr).zfill(4) + '.png'), I)

    @staticmethod
    def createMask(NOCSMap):
        LocNOCS = NOCSMap.type(torch.FloatTensor)

        Norm = torch.norm(LocNOCS, dim=0)
        ZeroT = torch.zeros((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device)
        OneT = torch.ones((1, LocNOCS.size(1), LocNOCS.size(2)), dtype=LocNOCS.dtype, device=LocNOCS.device) * 255 # Range: 0, 255
        Mask = torch.where(Norm >= 441.6729, ZeroT, OneT)  # 441.6729 == sqrt(3 * 255^2)
        return Mask.type(torch.FloatTensor)

    @staticmethod
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

    @staticmethod
    def convertData(RGB, Targets, isMaskNOX=False):
        TargetsTup = Targets
        Color00 = ptUtils.torch2np(RGB[0:3].squeeze()).squeeze() * 255

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
                    MaskedT, _ = GenericImageDataset.applyMask(torch.squeeze(T[0:4]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    OutTupRGB = OutTupRGB + (MaskedT,)
                else:
                    OutTupRGB = OutTupRGB + (ptUtils.torch2np(T[0:3]).squeeze(),)
                OutTupMask = OutTupMask + (ptUtils.torch2np(T[3]).squeeze(),)
            else:
                if isMaskNOX:
                    MaskedT0, _ = GenericImageDataset.applyMask(torch.squeeze(T[0:4]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    MaskedT1, _ = GenericImageDataset.applyMask(torch.squeeze(T[4:8]), Thresh=GenericImageDataset.L2MaskLoss.Thresh)
                    OutTupRGB = OutTupRGB + (MaskedT0, MaskedT1)
                else:
                    OutTupRGB = OutTupRGB + (ptUtils.torch2np(T[0:3]).squeeze(), ptUtils.torch2np(T[4:7]).squeeze())
                OutTupMask = OutTupMask + (ptUtils.torch2np(T[3]).squeeze(), ptUtils.torch2np(T[7]).squeeze())

        return Color00, OutTupRGB, OutTupMask

    def __init__(self, root, train=True, download=True, transform=None, target_transform=None,
                 imgSize=(640, 480), limit=100, loadMemory=False, FrameLoadStr=None, Required='color00', cameraIdx=-1):
        self.CameraIdx = cameraIdx
        self.nCameras = 10
        self.FileName = 'hand_rig_dataset_v3.zip'
        self.DataURL = 'https://storage.googleapis.com/stanford_share/Datasets/hand_rig_dataset_v3.zip'
        self.FrameLoadStr = ['color00', 'color01', 'normals00', 'normals01', 'nox00', 'nox01', 'pnnocs00', 'pnnocs01', 'uv00', 'uv01', 'camera'] if FrameLoadStr is None else FrameLoadStr
        self.LoadLevel = {}

        self.init(root, train, download, transform, target_transform, imgSize, limit, loadMemory, self.FrameLoadStr, Required)

        if 'camera' not in self.FrameLoadStr:
            print('[ WARN ]: Adding ''camera'' to FrameLoadStr to keep the dataloader happy.')
            self.FrameLoadStr.append('camera')

        self.loadData()

    def init(self, root, train=True, download=True, transform=None, target_transform=None, imgSize=(640, 480), limit=100, loadMemory=False, FrameLoadStr=None, Required='VertexColors'):
        self.DataDir = root
        self.isTrainData = train
        self.isDownload = download
        self.Transform = transform
        self.TargetTransform = target_transform
        self.ImageSize = imgSize
        self.LoadMemory = loadMemory
        self.Required = Required
        self.FrameLoadStr = FrameLoadStr
        if limit <= 0.0 or limit > 100.0:
            raise RuntimeError('Data limit percent has to be >0% and <=100%')
        self.DataLimit = limit

        if self.Required not in self.FrameLoadStr:
            raise RuntimeError('FrameLoadStr should contain {}.'.format(self.Required))

    def loadData(self):
        self.FrameFiles = {}
        # First check if unzipped directory exists
        DatasetDir = os.path.join(self.DataDir, os.path.splitext(self.FileName)[0])
        if os.path.exists(DatasetDir) == False:
            print("Dataset doesn't exist")
            exit()

        FilesPath = os.path.join(DatasetDir, 'val/')
        if self.isTrainData:
            FilesPath = os.path.join(DatasetDir, 'train/')

        CameraIdxStr = '*'
        if self.CameraIdx >= 0 and self.CameraIdx <= self.nCameras-1:
            CameraIdxStr = str(self.CameraIdx).zfill(2)

        print('[ INFO ]: Loading data for camera {}.'.format(CameraIdxStr))

        GlobPrepend = '_'.join(str(i) for i in self.FrameLoadStr)

        if CameraIdxStr == '*':
            GlobCache = os.path.join(FilesPath, 'all_glob_' + GlobPrepend + '.cache')
        else:
            GlobCache = os.path.join(FilesPath, 'glob_' + CameraIdxStr + '_' + GlobPrepend + '.cache')

        if os.path.exists(GlobCache):
            print('[ INFO ]: Loading from glob cache:', GlobCache)
            with open(GlobCache, 'rb') as fp:
                for Str in self.FrameLoadStr:
                    self.FrameFiles[Str] = pickle.load(fp)
        else:
            print('[ INFO ]: Saving to glob cache:', GlobCache)

            for Str in self.FrameLoadStr:
                self.FrameFiles[Str] = glob.glob(FilesPath + '/**/frame_*_view_' + CameraIdxStr + '_' + Str + '.*')
                self.FrameFiles[Str].sort()

            with open(GlobCache, 'wb') as fp:
                for Str in self.FrameLoadStr:
                    pickle.dump(self.FrameFiles[Str], fp)

        FrameFilesLengths = []
        for K, CurrFrameFiles in self.FrameFiles.items():
            if not CurrFrameFiles:
                raise RuntimeError('[ ERR ]: None data for', K)
            if len(CurrFrameFiles) == 0:
                raise RuntimeError('[ ERR ]: No files found during data loading for', K)
            FrameFilesLengths.append(len(CurrFrameFiles))

        if len(set(FrameFilesLengths)) != 1:
            raise RuntimeError('[ ERR ]: Data corrupted. Sizes do not match', FrameFilesLengths)

        TotSize = len(self)
        DatasetLength = math.ceil((self.DataLimit / 100) * TotSize)
        print('[ INFO ]: Loading {} / {} items.'.format(DatasetLength, TotSize))

        for K in self.FrameFiles:
            self.FrameFiles[K] = self.FrameFiles[K][:DatasetLength]

    def __getitem__(self, idx):
        RGB, LoadTup, Pose = self.loadImages(idx)
        LoadIms = torch.cat(LoadTup, 0)
        return RGB, (LoadIms, Pose)

    def loadImages(self, idx):
        Frame = {}

        for K in self.FrameFiles:
            if K == 'camera':
                with open(self.FrameFiles[K][idx], 'r') as JSONFile:
                    Frame[K] = json.load(JSONFile)
            else:
                Frame[K] = self.imread_rgb_torch(self.FrameFiles[K][idx], Size=self.ImageSize).type(torch.FloatTensor)
                if K != self.Required:
                    Frame[K] = torch.cat((Frame[K], self.createMask(Frame[K])), 0).type(torch.FloatTensor)
                if self.Transform is not None:
                    Frame[K] = self.Transform(Frame[K])

                # Convert range to 0.0 - 1.0
                Frame[K] /= 255.0

        # TODO
        # # Mask PNNOCS with the NOCS mask. This is a quirk with the data generation process
        # for K in self.FrameFiles:
        #     RepStr = K.replace('pnnocs', 'nox')
        #     if 'pnnocs' in K and RepStr in self.FrameFiles:
        #         # print('[ WARN ]: Replacing pnnocs mask with nox mask.', K, RepStr)
        #         Frame[K] = torch.cat((Frame[K][:3], Frame[RepStr][-1].unsqueeze(0)), 0).type(torch.FloatTensor)
        
        GroupedFrameStr = [list(i) for j, i in groupby(self.FrameLoadStr, lambda a: ''.join([i for i in a if not i.isdigit()]))]
        # print(self.FrameLoadStr)
        # print(GroupedFrameStr)

        LoadTup = ()
        # Concatenate any peeled outputs
        for Group in GroupedFrameStr:
            # print(Group)
            if 'camera' in Group[0]:
                continue
            Concated = ()
            for FrameStr in Group:
                # print(FrameStr)
                if 'color00' in FrameStr: # Append manually
                    continue
                Concated = Concated + (Frame[FrameStr],)
            if len(Concated) > 0:
                LoadTup = LoadTup + (torch.cat(Concated, 0), )

        # print(len(LoadTup))
        # print(LoadTup[0].size())

        return Frame['color00'], LoadTup, Frame['camera']

    def convertItem(self, idx, isMaskNOX=False):
        RGB, LoadTup, Pose = self.loadImages(idx)
        # RGB, Targets = self[idx]

        return self.convertData(RGB, LoadTup, isMaskNOX=isMaskNOX)
