# This file deals with handrgDataset, and make it operatble for NRNet, 
# which contains SegNet and IF-Net
# the result dataset would have a similar form with the handRigDataset

# NRNet dataset format:
# NRDataset
# +-- 0000
# ######## Origin Data ########
# |   + frame_00000000_NOCS_mesh.png
# |   + frame_00000000_view_00_color00.png
# |   + frame_00000000_view_00_color01.png
# |   + frame_00000000_view_00_normals00.png
# |   + frame_00000000_view_00_normals01.png
# |   + frame_00000000_view_00_nox00.png
# |   + frame_00000000_view_00_nox01.png
# |   + frame_00000000_view_00_pnnocs00.png
# |   + frame_00000000_view_00_pnnocs01.png
# |   + frame_00000000_view_00_uv00.png
# |   + frame_00000000_view_00_uv01.png
# ######## Data for IFNet ########
# |   ##### GT for IFNet #####
# |   + frame_00000000_isosurf_scaled.off
# |   + frame_00000000_boundary_0.1_samples.npz
# |   + frame_00000000_boundary_0.01_samples.npz
# |   + frame_00000000_transform.npz
# |   ##### Input for IFNet will be generated during pipeline
# +-- Other frame sets

import os,sys,glob,traceback,shutil
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
import argparse
import re
import implicit_waterproofing as iw

class DataFilter:

    def __init__(self,inputDir,outputDir):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.SampleNum = 100000
    def filter(self):
        
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)

        for mode in ['train','val','unseen_val']:
            CurOutDir = os.path.join(self.outputDir,mode)
            if not os.path.exists(CurOutDir):
                os.mkdir(CurOutDir)    
            
            self.mode = mode

            AllColorImgs = glob.glob(os.path.join(self.inputDir,mode,'**/frame_*_view_*_color00.*'))
            
            AllFrames = [self.findFrameNum(p) for p in AllColorImgs ]
            AllFrames = list(dict.fromkeys(AllFrames))
            AllFrames.sort()
            # AllFrames = ["00000000"]
        
            p = Pool(mp.cpu_count() >> 1)
            p.map(self.processFrame, AllFrames)
                            
    def processFrame(self,Frame):
        OutDir = os.path.join(self.outputDir,self.mode,str(int(Frame) // 100).zfill(4))
        if not os.path.exists(OutDir):
            try:
                os.mkdir(OutDir)
            except:
                print("subdir already exists")

        self.normalizeNOCS(Frame)
        for sigma in [0.01,0.1]:
            self.boudarySampling(Frame,sigma)
        self.copyImgs(Frame)

    def copyImgs(self,Frame):
        
        InDir = os.path.join(self.inputDir,self.mode,str(int(Frame) // 100).zfill(4))
        OutDir = os.path.join(self.outputDir,self.mode,str(int(Frame) // 100).zfill(4))
        ViewNum = 10
        for view in range(ViewNum):
            FrameView = "frame_" + Frame + "_view_" + str(view).zfill(2)
            Suffixs = ['_color00.png','_color01.png','_nox00.png','_nox01.png',\
                        '_pnnocs00.png','_pnnocs01.png',
                        '_normals00.png','_normals01.png','_uv00.png','_uv01.png']
            for fix in Suffixs:
                f_name = FrameView + fix
                old_f = os.path.join(InDir,f_name)
                new_f = os.path.join(OutDir,f_name)
                if os.path.exists(old_f) and not os.path.exists(new_f):
                    shutil.copy(old_f,new_f)


    def normalizeNOCS(self, Frame):
        
        InDir = os.path.join(self.inputDir,self.mode,str(int(Frame) // 100).zfill(4))
        OutDir = os.path.join(self.outputDir,self.mode,str(int(Frame) // 100).zfill(4))

        OriginMeshName = os.path.join(InDir,"frame_" + Frame + "_NOCS_mesh.obj")
        TargetMeshName = os.path.join(OutDir,"frame_" + Frame + "_isosurf_scaled.off")
        TransformName = os.path.join(OutDir,"frame_" + Frame + "_transform.npz")
        if os.path.exists(TargetMeshName):
            if args.write_over:
                print('overwrite ', TargetMeshName)
            else:
                print('File {} exists. Done.'.format(TargetMeshName))
                return

        translation = 0
        scale = 1

        try:
            mesh = trimesh.load(OriginMeshName, process=False)
            total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
            centers = (mesh.bounds[1] + mesh.bounds[0]) /2
            
            translation = -centers
            scale = 1/total_size

            mesh.apply_translation(-centers)
            mesh.apply_scale(1/total_size)
            mesh.export(TargetMeshName)
            
            np.savez(TransformName, translation=translation,scale=scale)
            print('Finished {}'.format(OriginMeshName))
        except Exception as e:
            print('Error {} with {}'.format(e,OriginMeshName))

        return 
    
    def boudarySampling(self,Frame,sigma):

        # Dir = os.path.dirname(ImgPath)
        # InDir = os.path.join(self.inputDir,self.mode,str(int(Frame) // 100).zfill(4))
        OutDir = os.path.join(self.outputDir,self.mode,str(int(Frame) // 100).zfill(4))

        MeshName = os.path.join(OutDir,"frame_" + Frame + "_isosurf_scaled.off")
        try:
            OutFile = os.path.join(OutDir,"frame_" + Frame + '_boundary_{}_samples.npz'.format(sigma))

            if os.path.exists(OutFile):
                if args.write_over:
                    print('overwrite ', OutFile)
                else:
                    print('File {} exists. Done.'.format(OutFile))
                    return

            mesh = trimesh.load(MeshName)
            points = mesh.sample(self.SampleNum)

            boundary_points = points + sigma * np.random.randn(self.SampleNum, 3)
            grid_coords = boundary_points.copy()
            grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

            grid_coords = 2 * grid_coords

            occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

            np.savez(OutFile, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
            print('Finished {}'.format(MeshName))
        except:
            print('Error with {}: {}'.format(MeshName, traceback.format_exc()))

    @staticmethod
    def findFrameNum(path):
        return re.findall(r'%s(\d+)' % "frame_",path)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='filter data for nrnet'
    )    
    parser.add_argument('--write-over',default=False,type=bool,help="Overwrite previous results if set to True")
    parser.add_argument('-o', '--output-dir',default='/ZG/meshedNOCS/IF_hand_dataset' ,help='Provide the output directory where the processed Model')
    parser.add_argument('-i', '--input-dir',default='/ZG/meshedNOCS/hand_rig_dataset_v3/', help='the root of dataset as input for nocs')
    args = parser.parse_args()

    df = DataFilter(inputDir=args.input_dir,outputDir=args.output_dir) 
    df.filter()