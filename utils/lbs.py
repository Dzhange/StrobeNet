import numpy as np
import torch
import torch.nn.functional as F

def lbs_(vertices, rot_mats, dtype=torch.float64):
    device = vertices.device
    batch_size = max(vertices.shape[0], rot_mats.shape[0])
    
    homogen_coord = torch.ones([batch_size, vertices.shape[1], 1],
                               dtype=dtype,
                               device=device)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)

    #     print(v_posed_homo)
    v_homo = v_posed_homo.clone()
    v_homo = torch.matmul(rot_mats, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts

def batch_inv(A, eps=1e-10):
    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone().data
    L = A.new_zeros(A.shape).data
    L[:, range(n), range(n)] = 1
    I = L.clone()

    # A = LU
    # [A I] = [LU I] -> [U L^{-1}]
    L_inv = I
    for i in range(n - 1):
        L[:, i + 1:, i:i + 1] = U[:, i + 1:, i:i + 1] / (U[:, i:i + 1, i:i + 1] + eps)
        L_inv[:, i + 1:, :] = L_inv[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(L_inv[:, i:i + 1, :])
        U[:, i + 1:, :] = U[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(U[:, i:i + 1, :])

    # [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
    A_inv = L_inv
    for i in range(n - 1, -1, -1):
        A_inv[:, i:i + 1, :] = A_inv[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)
        U[:, i:i + 1, :] = U[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)

        if i > 0:
            A_inv[:, :i, :] = A_inv[:, :i, :] - U[:, :i, i:i + 1].matmul(A_inv[:, i:i + 1, :])
            U[:, :i, :] = U[:, :i, :] - U[:, :i, i:i + 1].matmul(U[:, i:i + 1, :])

    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    
    rot_dir = rot_vecs / angle    

    # TODO: lbs for hand needs to be fixed
    # angle = np.deg2rad(angle) 
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()

     # relative joints position(as to parent)
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # get the transform for each joint
    # by combining rotation and translation
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    
    # transform_chain = [transforms_mat[:, 0]] #Add root transform
    
    bone_num = joints.shape[1]
    transform_chain = [0] * bone_num    
    
    # order for leapmotion hand
    # order = (0 ,1, 3, 2, 4, 13, 12, 5, 11, 10, 6, 9, 8, 7, 15, 14)
    order = (0, 1, 3, 2, 5, 13, 12, 4, 11, 10, 6, 9, 8, 7, 15, 14)
    transform_chain[0] = transforms_mat[:, 0]
    assert len(order) == bone_num
    for i in range(1, bone_num):
        index = order[i]
        curr_res = torch.matmul(transform_chain[parents[index]],
                                transforms_mat[:, index])
        transform_chain[index] = curr_res

    # if 0:
    #     transform_chain = [transforms_mat[:, 0]]
    #     for i in range(1, parents.shape[0]):
    #         # Subtract the joint location at the rest pose
    #         # No need for rotation, since it's identity when at rest
    #         curr_res = torch.matmul(transform_chain[parents[i]],
    #                                 transforms_mat[:, i])
    #         transform_chain.append(curr_res)
    # else:
    #     bone_num = joints.shape[1]
    #     got = np.array([0]*bone_num)
    #     got[0] = 1
    #     transform_chain = [0]*bone_num
    #     transform_chain[0] = transforms_mat[:, 0]
    #     while got.sum() != bone_num:
    #         for i in range(1,bone_num):
    #             # if parents isn't transformed
    #             if got[parents[i]] == 0:
    #                 continue
    #             # if parents has value
    #             elif got[i] == 0:
    #                 # set current joint valid
    #                 got[i] += 1
    #                 curr_res = torch.matmul(transform_chain[parents[i]],
    #                                 transforms_mat[:, i])
    #                 transform_chain[i] = curr_res
    #             else:
    #                 continue
    #     ##debug
    #     if 0:
    #         for i in range(bone_num):
    #             print(i, "th ")
    #             print(transforms_mat[:, i])
    #             print(transform_chain[i])
    
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

# def batch_rigid_transform_unity(rot_mats, rel_joints, dtype=torch.float32):

#     transforms_mat = transform_mat(
#         rot_mats.reshape(-1, 3, 3),
#         rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
#     transform_chain = []
#     for i in range(1, parents.shape[0]):
#         curr_res = torch.matmul(transform_chain[parents[i]],
#                                 transforms_mat[:, i])
#         transform_chain.append(curr_res)

def write_off(path, pc):
    import os
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a') as f1:
        for i in range(pc.shape[0]):
            p = pc[i]            
            f1.write("{} {} {}\n".format(p[0], p[1], p[2]))
    f1.close()

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row    
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def rotation_matrix(pose,
                    joints,                                        
                    lbs_weights,
                    parents,
                    pose2rot=True,
                    invert=True,
                    dtype=torch.float64):
    batch_size = pose.shape[0]
    device = pose.device
    
    # 3. Add pose blend shapes
    # N x J x 3 x 3    
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3),
                                   dtype=dtype).view([batch_size, -1, 3, 3])
        # (N x P) x (P, V * 3) -> N x V x 3
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)
    
    # print(rot_mats, rot_mats.shape)

    # 4. Get the global joint location    
    J_transformed, A = batch_rigid_transform(rot_mats, joints, parents, dtype=dtype)
    
    # print(J_transformed.shape, J_transformed)
    write_off("./J_trans.xyz", J_transformed[0].numpy())

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1]).type(dtype)
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = joints.shape[1]
    
    # print(A.shape) 1,2,4,4
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    T_inv = torch.zeros_like(T).to(device)

    if invert:
        for i in range(T.shape[0]):
            T_inv[i] = batch_inv(T[i])

    return T, T_inv