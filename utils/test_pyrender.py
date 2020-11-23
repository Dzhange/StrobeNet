
import trimesh
import cv2
import pyrender
import numpy as np 
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
fuze_trimesh = trimesh.load("D:/NRNet_expt/SAPIEN/compare/gls/occ_baseline_b1_100/frame_000_gt.off")
mesh = pyrender.mesh.Mesh.from_trimesh(fuze_trimesh)

scene = pyrender.Scene()
scene.add(mesh)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

s = np.sqrt(2)/2
camera_pose = np.array([
   [0.0, -s,   s,   0.3],
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  s,   s,   0.35],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
# r = pyrender.Viewer(400, 400)
color, depth = pyrender.Viewer(scene)
cv2.imwrite("pyrender.png", color)