import os, sys, glob
import argparse
import numpy as np
import trimesh
from utils.tools import voxels
from pykdtree.kdtree import KDTree

class evaluator():
    
    def __init__(self, args):
        self.args = args
        self.input_dir = args.input_dir
        self.load()

    def load(self):
        self.gt_occ_postfix = "gt.off"
        self.pred_occ_postfix = "recon.off"
        self.gt_occ_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.gt_occ_postfix)
        )
        self.pred_occ_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.pred_occ_postfix)
        )        
        assert len(self.gt_occ_paths) == len(self.pred_occ_paths)
        self.data_num = len(self.gt_occ_paths)

    def eval(self):        
        for i in range(self.data_num):
            print(i)
            gt_path = self.gt_occ_paths[i]
            pred_path = self.pred_occ_paths[i]

            gt_mesh = trimesh.load(gt_path, process=False)
            pred_mesh = trimesh.load(pred_path, process=False)

            p2s = self.p2s_dis(gt_mesh, pred_mesh)
            print(p2s)
            chamfer = self.chamfer_dis(gt_mesh, pred_mesh)
            print(chamfer)

    def p2s_dis(self, gt, pred):
        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        # completeness = trimesh.proximity.signed_distance(pred, gt.vertices)
        _, completeness, _ = pred.nearest.on_surface(gt.vertices)
        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        _, accuracy, _ = gt.nearest.on_surface(pred.vertices)
        p2s = (completeness**2).mean() + (accuracy**2).mean()

        return p2s

    def chamfer_dis(self, gt, pred):
        gt_pc = gt.vertices
        pred_pc = pred.vertices

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        kdtree = KDTree(pred_pc)
        completeness, _ = kdtree.query(gt_pc)
        
        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        kdtree = KDTree(gt_pc)
        dist, _ = kdtree.query(pred_pc)
        accuracy = (dist**2).mean()
        
        chamfer = (completeness**2 + accuracy**2).mean()
        return chamfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', '--input-dir', help='Specify the location of the directory of data to evaluate', required=True)
    args, _ = parser.parse_known_args()

    eval = evaluator(args)
    eval.eval()