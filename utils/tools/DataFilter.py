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

import os, sys, glob, traceback, shutil
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import shutil
import argparse
import re
import implicit_waterproofing as iw

ERROR = -1

class DataFilter:

    def __init__(self, args):
        self.inputDir = args.input_dir
        self.outputDir = args.output_dir
        self.sapien = args.sapien
        self.pose_num = args.pose_per_actor

        self.SampleNum = 100000
        if self.sapien:
            self.frame_per_dir = self.pose_num
        else:
            self.frame_per_dir = 100

    def filter(self):        
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)

        
        if self.sapien:
            config_path = os.path.join(self.outputDir, "config.json")
            if os.path.exists(config_path):
                os.remove(config_path)
                
            f = open(config_path, "a")
            configs = '{' + "\"pose_num\": {}".format(self.pose_num) + '}'
            f.write(configs)
            f.close()

        for mode in ['train', 'val']:
            cur_out_dir = os.path.join(self.outputDir,mode)
            if not os.path.exists(cur_out_dir):
                os.mkdir(cur_out_dir)
            
            
            if 0:
                self.mode = mode
            
                all_color_imgs = glob.glob(os.path.join(self.inputDir, mode, '**/frame_*_view_*_color00.*'))
                all_color_imgs = glob.glob(os.path.join(self.inputDir, mode, '**/frame_*_color00.*'))
                all_frames = [self.findFrameNum(p) for p in all_color_imgs ]
                all_frames = list(dict.fromkeys(all_frames))
                all_frames.sort()
            
                p = Pool(mp.cpu_count()>>3)            
                p.map(self.processFrame, all_frames)
            else:
                self.mode = "val"
                self.processFrame("00048000")

    def processFrame(self, Frame):
        out_dir = os.path.join(self.outputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                print("subdir already exists")
        success = self.normalizeNOCS(Frame)
        if success == ERROR:
            return
        for sigma in [0.01, 0.1]:
            self.boudarySampling(Frame, sigma)
        self.copyImgs(Frame)
        return

    def normalizeNOCS(self, Frame):

        in_dir = os.path.join(self.inputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        out_dir = os.path.join(self.outputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        if self.sapien:
            Frame = str(int(Frame) // self.pose_num * self.pose_num).zfill(8)
            orig_mesh_path = os.path.join(in_dir, "frame_" + Frame + "_wt_mesh.obj")
        else:
            orig_mesh_path = os.path.join(in_dir, "frame_" + Frame + "_NOCS_pn_mesh.obj")
        target_mesh_path = os.path.join(out_dir, "frame_" + Frame + "_isosurf_scaled.off")
        transform_path = os.path.join(out_dir, "frame_" + Frame + "_transform.npz")
        if os.path.exists(target_mesh_path):
            if args.write_over:
                print('overwrite ', target_mesh_path)
            else:
                # print('File {} exists. Done.'.format(target_mesh_path))
                return 0

        translation = 0
        scale = 1
        try:
            mesh = trimesh.load(orig_mesh_path, process=False)
            total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
            centers = (mesh.bounds[1] + mesh.bounds[0]) /2
            translation = -centers
            scale = 1/total_size
            mesh.apply_translation(-centers)
            mesh.apply_scale(1/total_size)
            mesh.export(target_mesh_path)
            np.savez(transform_path, translation=translation, scale=scale)
            # print('Finished {}'.format(orig_mesh_path))
        except Exception as e:
            print('Error normalize_NOCS {} with {}'.format(e, orig_mesh_path))
            return -1
        return 0

    def boudarySampling(self, Frame, sigma):
        out_dir = os.path.join(self.outputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        if self.sapien:
            Frame = str(int(Frame) // self.pose_num * self.pose_num).zfill(8)

        mesh_path = os.path.join(out_dir, "frame_" + Frame + "_isosurf_scaled.off")
        while not os.path.exists(mesh_path):
            print("waiting for ", mesh_path)
            continue
        try:
            out_file = os.path.join(out_dir, "frame_" + Frame + '_boundary_{}_samples.npz'.format(sigma))

            if os.path.exists(out_file):
                if args.write_over:
                    print('overwrite ', out_file)
                else:
                    # print('File {} exists. Done.'.format(out_file))
                    return

            mesh = trimesh.load(mesh_path)
            points = mesh.sample(self.SampleNum)

            boundary_points = points + sigma * np.random.randn(self.SampleNum, 3)
            grid_coords = boundary_points.copy()
            grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

            grid_coords = 2 * grid_coords

            occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

            np.savez(out_file, points=boundary_points, occupancies=occupancies, grid_coords=grid_coords)
            # print('Finished {}'.format(mesh_path))
        except:
            print('Error with {}: {}'.format(mesh_path, traceback.format_exc()))

    def copyImgs(self, Frame):
        in_dir = os.path.join(self.inputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        out_dir = os.path.join(self.outputDir, self.mode, str(int(Frame) // self.frame_per_dir).zfill(4))
        if self.sapien:
            view = 0            
            frame_view = "frame_" + Frame
            suffixs = ['_view_00_color00.png', '_view_00_nox00.png', '_view_00_pnnocs00.png',\
                 '_view_00_linkseg.png', "_pose.txt", '_wt_mesh.obj']
            for fix in suffixs:
                f_name = frame_view + fix
                old_f = os.path.join(in_dir, f_name)                
                new_f = os.path.join(out_dir, f_name)
                if os.path.exists(old_f) and not os.path.exists(new_f):
                    shutil.copy(old_f, new_f)
        else:
            view_num = 10
            for view in range(view_num):
                frame_view = "frame_" + Frame + "_view_" + str(view).zfill(2)
                suffixs = ['_color00.png','_color01.png','_nox00.png','_nox01.png',\
                            '_pnnocs00.png','_pnnocs01.png',
                            '_normals00.png','_normals01.png','_uv00.png','_uv01.png']
                for fix in suffixs:
                    f_name = frame_view + fix
                    old_f = os.path.join(in_dir, f_name)
                    new_f = os.path.join(out_dir, f_name)
                    if os.path.exists(old_f) and not os.path.exists(new_f):
                        shutil.copy(old_f, new_f)

    @staticmethod
    def findFrameNum(path):
        return re.findall(r'%s(\d+)' % "frame_",path)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='filter data for nrnet'
    )    
    parser.add_argument('--write-over', default=False, type=bool, help="Overwrite previous results if set to True")
    parser.add_argument('-o', '--output-dir', default='/ZG/meshedNOCS/IF_hand_dataset' ,help='Provide the output directory where the processed Model')
    parser.add_argument('-i', '--input-dir', default='/ZG/meshedNOCS/hand_rig_dataset_v3/', help='the root of dataset as input for nocs')
    parser.add_argument('-s','--sapien', default=False, type=bool, help="SAPIEN dataset is a little different")
    # args = parser.parse_args()
    parser.add_argument('-p', '--pose_per_actor', type=int, required=True, help="if use sapien, must specify number of poses")
    args = parser.parse_args()

    df = DataFilter(args) 
    df.filter()