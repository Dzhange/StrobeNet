import os, sys, glob
import cv2
import argparse
import numpy as np
import open3d as o3d
import pyrender
import trimesh

def render_pc(pc, output_path):
    """
    render a image from an numpy pc input of shape (N, 3)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=400,height=400,visible=False)
    # fronts = [(0,0,1),(0,1.0,1e-6),(1.0,1e-6,0)]

    # nocs_color = nocs #just color nocs with nocs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pc))
    
    # print(pcd.points)
    # print(pcd.colors)
    # for j, front in enumerate(fronts):                
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    # ctr.change_field_of_view(1)
    vis.poll_events()
    vis.update_renderer()            
    vis.capture_screen_image(output_path)
    vis.remove_geometry(pcd)


class Visualizer(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        self.baseline_dir = args.baseline_dir
        self.output_dir = args.output_dir
        
        self.frames = args.frames
        self.view_num = args.view_num

        if os.path.exists(self.output_dir) == False:
            os.mkdir(self.output_dir)
        self.load()

    def visualize(self):
        """
        generate a very large image
        each row is an instance
        each column is a different case
        """
        self.data_type = [ "baseline", "ours", "gt"]
        self.render_mesh(self.gt_paths, "gt")
        self.render_mesh(self.result_paths, "ours")
        self.render_mesh(self.baseline_paths, "baseline")

        # self.render_nocs(self.gt_nocs_paths, "gt_nocs")
        # self.render_nocs(self.pred_nocs_paths, "pred_nocs")

        self.cat()

    def load(self):
        """
        load necessary files
        """
        
        # postfix = "obj"
        self.gt_occ_postfix = "gt.obj"
        self.pred_occ_postfix = "recon.obj"
        self.bl_occ_postfix = "recon.off"
        # self.bl_occ_postfix = "recon.obj"
        
        self.gt_nocs_postfix = "gt.png"
        self.pred_nocs_postfix = "pred.png"


        self.gt_paths = []
        self.result_paths = []
        self.baseline_paths = []
        self.frame_imgs = []

        for frame in self.frames:
            gt_file_name = "frame_{}_{}".format(str(frame).zfill(3), self.gt_occ_postfix)
            pred_file_name = "frame_{}_{}".format(str(frame).zfill(3), self.pred_occ_postfix)
            bl_file_name = "frame_{}_{}".format(str(frame).zfill(3), self.bl_occ_postfix)

            self.gt_paths.append(
                os.path.join(self.result_dir, gt_file_name)
            )
            self.result_paths.append(
                os.path.join(self.result_dir, pred_file_name)
            )
            self.baseline_paths.append(
                os.path.join(self.baseline_dir,bl_file_name)
            )            
            
            img_file_name = ["frame_{}_view_{}_color00.png".format(str(frame).zfill(3), vi) for vi in range(self.view_num) ] 
            img_path = [os.path.join(self.result_dir, img_fn) for img_fn in img_file_name]
            self.frame_imgs.append(
                img_path
            )


        assert len(self.gt_paths)\
                == len(self.result_paths)\
                == len(self.baseline_paths)\
                == len(self.frame_imgs)

        self.data_num = len(self.gt_paths)


    def filter(self):
        """
        basically just reverse the index of the isosurf mesh
        """        
        for path in self.gt_paths:
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
        # self.fronts = [(0, 0, 1),(0, 1.0, 1e-6),(1.0, 1e-6, 0)]
        # self.fronts = [(0,1,0.0001)]
        self.fronts = [(0.3,-0.7,0.8001)]
        for i, path in enumerate(paths):
            mesh = trimesh.load(path)
            trimesh.repair.fix_inversion(mesh)
            mesh.export(path.replace("off", "obj"))

            mesh = o3d.io.read_triangle_mesh(path.replace("off", "obj"))
            mesh.compute_vertex_normals()

            # normals = np.asarray(mesh.vertex_normals)
            # nm = np.linalg.norm(normals, axis=1)
            # normals /= nm[:, np.newaxis]
            # # normals *= 10
            # mesh.vertex_colors = o3d.utility.Vector3dVector(normals)
            
            
            for j, front in enumerate(self.fronts):
                vis.add_geometry(mesh)  

                ctr = vis.get_view_control()

                # parameters = o3d.io.read_pinhole_camera_parameters("gls_new.json")
                parameters = o3d.io.read_pinhole_camera_parameters("laptop.json")
                # parameters = o3d.io.read_pinhole_camera_parameters("oven.json")
                ctr.convert_from_pinhole_camera_parameters(parameters)
                
                vis.poll_events()
                vis.update_renderer()
# python ..\Code\StrobeNet\vis\render_stich.py -rd gls_2v -bd gls_onet_v2 -o fig_4 -f  233 322 -cn False -v 2
                output_path = os.path.join(self.output_dir,\
                                "{}_{}_{}.png".format(name, self.frames[i], j))
                vis.capture_screen_image(output_path)
                vis.remove_geometry(mesh)
        vis.destroy_window()

    def render_nocs(self,paths, name):
        """
        lifts nocs map into 3d pointcloud and then render
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=400,height=400,visible=False)
        # fronts = [(0,0,1),(0,1.0,1e-6),(1.0,1e-6,0)]
        for i, path in enumerate(paths):
            nocs = self.nocs2pc(path)
            # nocs_color = nocs #just color nocs with nocs
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(nocs)
            pcd.colors = o3d.utility.Vector3dVector(nocs)

            # for j, front in enumerate(fronts):                
            vis.add_geometry(pcd)                

            ctr = vis.get_view_control()
            # ctr.change_field_of_view(1)
            vis.poll_events()
            vis.update_renderer()
            output_path = os.path.join(self.output_dir,\
                            "{}_{}_{}.png".format(name, i, j))
            vis.capture_screen_image(output_path)
            vis.remove_geometry(pcd)
    
    def cat_(self):
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
    
    def cat(self):
        for i in range(self.data_num):
            # each frame of one type of data forms one row            
            row_to_cat = ()
            for data in self.data_type:
                for ft in range(len(self.fronts)):
                    file_name = "{}_{}_{}.png".format(data, self.frames[i], ft)

                    path = os.path.join(self.output_dir, file_name)
                    img = cv2.imread(path)
                    row_to_cat = row_to_cat + (img, )
            
            colors = ()
            for vi in range(self.view_num):
                # print(self.frame_imgs[i][vi])
                color = cv2.imread(
                    self.frame_imgs[i][vi]
                )
            
                height = 400
                width = int(color.shape[1] * (height / color.shape[0]))
            
                color = cv2.resize(color, (width,height))
                colors = colors + (color, )
            
            row_to_cat = colors + row_to_cat
            
            frame = np.concatenate(row_to_cat, axis=1)
            
            cat_path = os.path.join(
                    self.output_dir,                    
                    "frame_{}_all.png".format(i)
                )       
            # print(cat_path)
            # cv2.imwrite(cat_path, frame)
            cv2.imwrite(cat_path, frame)


    def nocs2pc(self, path):
        nocs_map = cv2.imread(path)
        nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
        valid_idx = np.where(np.all(nocs_map != [255, 255, 255], axis=-1)) # Only white BG                    
        valid_points = nocs_map[valid_idx[0], valid_idx[1]] / 255
        return valid_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--flip', help='if set to true, the fliper would flip the face orientation for the gt mesh using meshlab', type=bool, default=False)
    # parser.add_argument('-i', '--input-dir', help='Specify the location of the directory of data to visualize', required=True)
    parser.add_argument('-rd', '--result-dir', type=str, help='data dir of our result', required=True)    
    parser.add_argument('-bd', '--baseline-dir', type=str, help='data dir of baseline result', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='dir for output', required=True)

    parser.add_argument('-f', '--frames', nargs="+", type=int, help='order of frames', required=True)
    parser.add_argument('-v', '--view_num', type=int, help='order of frames', required=True)
    parser.add_argument('-cn', '--color_normal', type=bool, help='color mesh with normals', required=True)


    args, _ = parser.parse_known_args()
    dv = Visualizer(args)
    # if args.flip == True:
    #     dv.filter()
    dv.visualize()
    dv.cat()