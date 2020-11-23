import os, sys, glob
import cv2
import argparse
import numpy as np
import open3d as o3d
import trimesh



# vis = o3d.visualization.Visualizer()
# vis.create_window(width=400,height=400,visible=True)
# #don't know why, but seems this direction needs a tiny distortion
# # self.fronts = [(0, 0, 1),(0, 1.0, 1e-6),(1.0, 1e-6, 0)]
# f = (1, 1, 0.0001)
# self.fronts = [(0.3,-0.7,0.8001)]

# mesh = o3d.io.read_triangle_mesh("D:/NRNet_expt/SAPIEN/compare/gls/occ_baseline_b1_100/frame_000_gt.off")
mesh = o3d.io.read_triangle_mesh("d:/GuibasLab/FIGURES/laptop_strobe_v2/frame_000_gt.off")
# mesh = o3d.io.read_triangle_mesh("d:/GuibasLab/FIGURES/gls_2v/frame_200_gt.obj")

# mesh = o3d.io.read_triangle_mesh(path)
mesh.compute_vertex_normals()

normals = np.asarray(mesh.vertex_normals)
nm = np.linalg.norm(normals, axis=1)
normals /= nm[:, np.newaxis]
# normals *= 10
# mesh.vertex_colors = o3d.utility.Vector3dVector(normals)

vis = o3d.visualization.Visualizer()
vis.create_window(width=400,height=400)
vis.add_geometry(mesh)

ro = vis.get_render_option()
ro.light_on = True

# vis.get_render_option().load_from_json("../../TestData/renderoption.json")
vis.run()
vis.destroy_window()

# o3d.visualization.draw_geometries([mesh,])
# vis.update_renderer()
        
#         output_path = os.path.join(self.output_dir,\
#                         "{}_{}_{}.png".format(name, self.frames[i], j))
#         vis.capture_screen_image(output_path)
#         vis.remove_geometry(mesh)
# vis.destroy_window()