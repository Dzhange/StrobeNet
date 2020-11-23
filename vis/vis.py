import os, sys, glob
import cv2
import argparse
import numpy as np
import open3d as o3d
import trimesh

class DataVisualizer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = os.path.join(self.input_dir,'vis')
        if os.path.exists(self.output_dir) == False:
            os.mkdir(self.output_dir)
        self.load()
    
    def visualize(self):
        """
        generate a very large image
        each row is an instance
        each column is a different case
        """
        self.render_mesh(self.gt_occ_paths, "gt_occ")
        self.render_mesh(self.pred_occ_paths, "pred_occ")
        self.render_nocs(self.gt_nocs_paths, "gt_nocs")
        self.render_nocs(self.pred_nocs_paths, "pred_nocs")

        self.cat()

    def load(self):
        """
        load necessary files
        """
        self.gt_occ_postfix = "gt.off"
        self.pred_occ_postfix = "recon.off"
        
        self.gt_nocs_postfix = "gt.png"
        self.pred_nocs_postfix = "pred.png"

        self.gt_occ_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.gt_occ_postfix)
        )
        self.pred_occ_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.pred_occ_postfix)
        )
        self.gt_nocs_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.gt_nocs_postfix)
        )
        self.pred_nocs_paths = glob.glob(
            os.path.join(self.input_dir, "*" + self.pred_nocs_postfix)
        )

        assert len(self.gt_nocs_paths)\
                == len(self.pred_nocs_paths)\
                == len(self.gt_occ_paths)\
                == len(self.pred_occ_paths)

        self.data_num = len(self.pred_occ_paths)
        

    def filter(self):
        """
        basically just reverse the index of the isosurf mesh
        """        
        for path in self.gt_occ_paths:
            mesh = trimesh.load(path)
            trimesh.repair.fix_inversion(mesh)
            mesh.export(path)

    def render_mesh(self, paths, name):
        """
        input: paths of mesh to render
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=400,height=400,visible=False)
        #don't know why, but seems this direction needs a tiny distortion
        fronts = [(0, 0, 1),(0, 1.0, 1e-6),(1.0, 1e-6, 0)]
        for i, path in enumerate(paths):
            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            ctr = vis.get_view_control()
            for j, front in enumerate(fronts):
                vis.add_geometry(mesh)
                ctr.set_front(front)
                vis.poll_events()
                vis.update_renderer()
                output_path = os.path.join(self.output_dir,\
                                "{}_{}_{}.png".format(name, i, j))
                vis.capture_screen_image(output_path)
                vis.remove_geometry(mesh)
        vis.destroy_window()
    
    def render_nocs(self,paths, name):
        """
        lifts nocs map into 3d pointcloud and then render
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=400,height=400,visible=False)
        fronts = [(0,0,1),(0,1.0,1e-6),(1.0,1e-6,0)]
        for i, path in enumerate(paths):
            nocs = self.nocs2pc(path)
            # nocs_color = nocs #just color nocs with nocs
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(nocs)
            pcd.colors = o3d.utility.Vector3dVector(nocs)
            ctr = vis.get_view_control()
            for j, front in enumerate(fronts):                
                vis.add_geometry(pcd)
                ctr.set_front(front)                
                vis.poll_events()
                vis.update_renderer()    
                output_path = os.path.join(self.output_dir,\
                                "{}_{}_{}.png".format(name, i, j))
                vis.capture_screen_image(output_path)
                vis.remove_geometry(pcd)

    def cat(self):
        # BigImg = ()
        for i in range(self.data_num):
            # each frame of one type of data forms one row            
            row_to_cat = ()
            for data in ['nocs', 'occ']:
                for view in range(3):
                    for mode in ['gt', 'pred']:                    
                        file_name = "{}_{}_{}_{}.png".format(mode, data, i, view)
                        path = os.path.join(self.output_dir, file_name)
                        img = cv2.imread(path)
                        row_to_cat = row_to_cat + (img, )
            color = cv2.imread(
                os.path.join(
                    self.input_dir,
                    "frame_{}_color00.png".format(str(i).zfill(3))
                )
            )            
            height = 400
            width = int(color.shape[1] * (height / color.shape[0]))
            color = cv2.resize(color, (width,height))            
            row_to_cat = (color,) + row_to_cat                    
            frame = np.concatenate(row_to_cat, axis=1)
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    "sums",
                    "frame_{}_all.png".format(i)
                ),
                frame
            )            

    def nocs2pc(self, path):
        nocs_map = cv2.imread(path)
        nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
        valid_idx = np.where(np.all(nocs_map != [255, 255, 255], axis=-1)) # Only white BG                    
        valid_points = nocs_map[valid_idx[0], valid_idx[1]] / 255
        return valid_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--flip', help='if set to true, the fliper would flip the face orientation for the gt mesh using meshlab', type=bool, default=False)
    parser.add_argument('-i', '--input-dir', help='Specify the location of the directory of data to visualize', required=True)
    args, _ = parser.parse_known_args()

    dv = DataVisualizer(args)

    if args.flip == True:
        dv.filter()
    # dv.visualize()
    dv.cat()