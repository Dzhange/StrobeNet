from tools.voxels import VoxelGrid
import tools.implicit_waterproofing as iw
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

bb_min = -0.5
bb_max = 0.5

grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, 128)
kdtree = KDTree(grid_points)

off_path = "/workspace/Data/SAPIEN/debug/wt_trs_IF/train/0000/frame_00000000_isosurf_scaled.off"
mesh = trimesh.load(off_path)
point_cloud = mesh.sample(3000)
# print(point_cloud.shape)
point_cloud = point_cloud + 0.01 * np.random.randn(point_cloud.shape[0], 3)
occupancies = np.zeros(len(grid_points), dtype=np.int8)

_, idx = kdtree.query(point_cloud)
occupancies[idx] = 1
occupancies = np.reshape(occupancies, (128,)*3)
VoxelGrid(occupancies, (0, 0, 0), 1).to_mesh().export("/workspace/test.off")