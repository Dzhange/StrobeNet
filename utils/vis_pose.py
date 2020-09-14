import os
import open3d as o3d
import numpy as np
import argparse


show_frame = True
cap = True

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    qTrans_Mat = np.eye(3,3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

def create_arrow_my(scale=1):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.02
    cylinder_height = scale*0.08
    cone_radius = scale/100
    cylinder_radius = scale/200

    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    
    return mesh_frame

def get_arrow_my(orig, vec):
    mesh_arrow = create_arrow_my(1)
    rot_mat = caculate_align_mat(vec)
    mesh_arrow.rotate(rot_mat, center=[0,0,0])
    mesh_arrow.translate(orig)
    return mesh_arrow

def render(mesh_file, pose_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    print(mesh_file)
    frame = o3d.geometry.LineSet()
    frame = frame.create_from_triangle_mesh(mesh)

    pose = np.loadtxt(pose_file)
    location = pose[:, 0:3]
    direction = pose[:, 3:6]
    parents = pose[:, 6]

    line = []
    for i, p in enumerate(parents):
        if p >= 0:
            line.append([i,p])

    skeleton = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(location),
        lines=o3d.utility.Vector2iVector(line)
    )

    bones = []
    # convert all joint directions into arrows for vis
    for i in range(pose.shape[0]):        
        scale = np.linalg.norm(direction[i])
        vec = direction[i] / scale
        
        bones.append(            
            get_arrow_my(location[i], vec=vec)
        )

    output_dir = "./"    
    debug = 0
    if not debug:
        vis = o3d.visualization.Visualizer()
        # vis.create_window(width=400,height=400,visible=False)
        vis.create_window(width=400,height=400,visible=True)

        fronts = [(0, 0, 1),(0, 1.0, 1e-6),(1.0, 1e-6, 0)]
        ctr = vis.get_view_control()
        
        if show_frame:
            vis.add_geometry(frame)

        vis.add_geometry(skeleton)
        # for bone in bones:
            # vis.add_geometry(bone)

        
        if cap:
            for j, front in enumerate(fronts):            
                ctr.set_front(front)
                vis.poll_events()
                vis.update_renderer()

                output_path = os.path.join(
                                output_dir,
                                "{}.png".format(j))
                vis.capture_screen_image(output_path)
        else:
            vis.run()        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NOCSMapModule to visualize NOCS maps and camera poses.', fromfile_prefix_chars='@')
    arg_group = parser.add_argument_group()    
    arg_group.add_argument('--path', '-p', help='path to the pose file to be visualized', required=False, default=None)

    args, _ = parser.parse_known_args()
    
    # pose_file = args.path
    
    nocs = 0
    # show_frame = False
    cap = False

    if nocs:
        pose_file = "D:/GuibasLab/Data/pose/0000/frame_00000000_hpose_nocs.txt"
        mesh_file = pose_file.replace('hpose_nocs.txt', 'NOCS_mesh.obj')
    else:
        pose_file = "D:/GuibasLab/Data/pose/0000/frame_00000000_hpose_glob.txt"
        mesh_file = pose_file.replace('hpose_glob.txt', 'World_mesh.obj')

    

    render(mesh_file, pose_file)