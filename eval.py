import os, sys, glob
import argparse
import numpy as np
import trimesh
from utils.tools import voxels
from utils.tools.implicit_waterproofing import implicit_waterproofing
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
        self.gt_occ_paths.sort()
        self.pred_occ_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.pred_occ_postfix)
        )       
        self.pred_occ_paths.sort()

        assert len(self.gt_occ_paths) == len(self.pred_occ_paths)
        self.data_num = len(self.gt_occ_paths)

    def eval(self):

        p2s_dis_list = [0]
        chamfer_dis_list = [0]
        iou_list = []
        for i in range(self.data_num):            
            gt_path = self.gt_occ_paths[i]
            pred_path = self.pred_occ_paths[i]
            print(pred_path)
            if 1:
                os.system("/workspace/Manifold/build/simplify -i {} -o {} -f {}".format(gt_path, gt_path, 5000))
                os.system("/workspace/Manifold/build/simplify -i {} -o {} -f {}".format(pred_path, pred_path, 5000))

            # print("gt_path ", gt_path)
            gt_mesh = trimesh.load(gt_path, process=False)
            pred_mesh = trimesh.load(pred_path, process=False)
            
            p2s = self.p2s_dis(gt_mesh, pred_mesh)
            p2s = np.sqrt(p2s)
            print(p2s)
            p2s_dis_list.append(p2s)
            
            chamfer = self.chamfer_dis(gt_mesh, pred_mesh)
            chamfer = np.sqrt(chamfer)
            chamfer_dis_list.append(chamfer)
            print(chamfer)
            
            iou = self.iou(gt_mesh, pred_mesh)
            print("current iou ", iou)
            iou_list.append(iou)


            done = int(30 * (i+1) / self.data_num)
            sys.stdout.write(('\r[{}>{}] p2s dis - {:.8f} chamfer dis - {:.8f} IoU - {:.8f}' )
                             .format('+' * done, '-' * (30 - done), \
                                np.mean(np.asarray(p2s_dis_list)), \
                                np.mean(np.asarray(chamfer_dis_list)),\
                                np.mean(np.asarray(iou_list)),\
                                ))
            sys.stdout.flush()

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

    def iou(self, mesh_gt, mesh_pred):
        bb_max = 0.5
        bb_min = -0.5
        n_points=100000

        bb_len = bb_max - bb_min
        bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

        occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
        occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

        area_union = (occ_pred | occ_gt).astype(np.float32).sum()
        area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

        iou = (area_intersect / area_union)
        
        return iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', '--input-dir', help='Specify the location of the directory of data to evaluate', required=True)
    args, _ = parser.parse_known_args()

    eval = evaluator(args)
    eval.eval()