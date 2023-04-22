import os
import json
import argparse

import imageio
import numpy as np
import trimesh
import trimesh.transformations as tt
import open3d as o3d
from tqdm import tqdm


listfiles = lambda root : \
            [os.path.join(base, f) 
                for base, _, files in os.walk(root) 
                    if files for f in files
            ]


def poisson_o3d(
    pcd: o3d.geometry.PointCloud,
    depth: int=10,
):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        (
            mesh_poisson,
            densities,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh_poisson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        required=True,
        help="path to the propcess direction",
    )
    args = parser.parse_args()

    print("Post processing from colmap")
    work_dir = args.work_dir
    assert os.path.exists(work_dir), "Path dose not exist"


    # Load all pose files:
    pose_origin_files = listfiles(
        os.path.join(
            work_dir, "pose_origin"
        )
    )
    pose_origin_files.sort(key=lambda x: int(os.path.basename(x)[:-4].split("_")[1]))
    poses_origin = np.stack([np.loadtxt(posf) for posf in pose_origin_files])
    pose_trans_path = os.path.join(work_dir, "pose")
    os.makedirs(pose_trans_path, exist_ok=True)

    # Load Intrinsic 
    intrinsic_file = os.path.join(
        work_dir, "transforms_origin.json"
    )
    with open(intrinsic_file, "r", encoding="utf-8") as f:
        camera_info = json.load(f)
    fx = camera_info['fl_x']
    fy = camera_info['fl_y']
    cx = camera_info['cx']
    cy = camera_info['cy']
    height = camera_info['h']
    width = camera_info['w']

    intrinsic = np.eye(4)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    np.savetxt(f"{work_dir}/intrinsic.txt", intrinsic)


    yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
    )
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz =  np.ones_like(xx)

    dirs = np.stack((xx, yy, zz), axis=-1)  # OpenCV convention
    dirs_ = dirs.reshape(-1, 3)  # [H * W, 3]

    # Get path to obj pts
    colmap_sfm_pts = os.path.join(work_dir, "sparse_sfm_points.ply")
    colmap_mvs_pts = os.path.join(work_dir, "colmap/fuse.ply")
    
    sparse_or_dense = 'dense'
    if sparse_or_dense == 'dense':
        # ###########################################################################
        # ''' Step one: reconstruct the mesh and manually clean it '''
        # pcd = o3d.io.read_point_cloud(colmap_mvs_pts)
        # poisson_mesh = poisson_o3d(pcd, depth=9)
        # o3d.io.write_triangle_mesh(os.path.join(work_dir, f"mvs_colmap_poisson.ply"), poisson_mesh)

        # msg = input('Clean the mesh...[y/n]') # observe pose file size
        # if msg != 'y':
        #     exit()

        # Mesh from colmap to opencv coornidate
        # ref_mesh = trimesh.load(os.path.join(work_dir, f"mvs_colmap_poisson.ply"))
        # nvertices = np.asarray(ref_mesh.vertices)
        # nvertices = nvertices[:, np.array([1,0,2])]
        # nvertices[..., 2] *= -1
        # ref_mesh.vertices = nvertices
        # ref_mesh.export(os.path.join(work_dir, f"mvs_opencv_poisson.ply"))

        ###########################################################################
        ''' Step two: scale the object into normalized bbox'''
        scale = 1; offset = np.array([0,0,0])
        ref_mesh = trimesh.load(os.path.join(work_dir, f"mvs_opencv_poisson.ply"))
        min_bound, max_bound = np.array(ref_mesh.bounds)
        offset = (max_bound + min_bound) / 2
        scale = 1.86 / (max_bound - min_bound).max()  # within bound [-0.93, 0.93]
        norm_vertices = np.asarray(ref_mesh.vertices)
        norm_vertices -= offset
        norm_vertices *= scale
        ref_mesh.vertices = norm_vertices
        ref_mesh.export(os.path.join(work_dir, f"mvs_opencv_poisson_norm.ply"))

        poses_norm = []
        for pose_ori in poses_origin:
            rescale_pose = np.eye(4)
            rescale_pose[:3, :3] = pose_ori[:3, :3]
            rescale_pose[:3, 3] = (pose_ori[:3, 3] - offset) * scale
            poses_norm.append(rescale_pose)
        poses_norm = np.stack(poses_norm)


        ###########################################################################
        ''' Step three: rot the object to be z-axis up, and xy plane as ground'''
        norm_mesh = trimesh.load(os.path.join(work_dir, "mvs_opencv_poisson_norm.ply"))
        # rxyz = tt.euler_matrix(0 / 180 * np.pi, 30 / 180 * np.pi, 0 / 180 * np.pi, "rxyz")
        # rlist = [0.634894, 0.39075, 0.666501, 0, -0.343688, 0.915459, -0.209318, 0, -0.691945, -0.0961734, 0.715516, 0, 0, 0, 0, 1 ]
        rlist = [0.607278, 0.451851, 0.653487, 0, -0.399084, 0.884712, -0.240866, 0, -0.686983, -0.114524, 0.717592, 0, 0, 0, 0, 1]
        rxyz=np.array(rlist).reshape(4,4)

        nvertices = np.asarray(norm_mesh.vertices)
        nvertices = nvertices @ rxyz[:3, :3].T
        norm_mesh.vertices = nvertices
        norm_mesh.export(os.path.join(work_dir, f"mvs_opencv_poisson_norm_zup.ply"))

        # Scale again
        scale = 1; offset = np.array([0,0,0])
        ref_mesh = trimesh.load(os.path.join(work_dir, f"mvs_opencv_poisson_norm_zup.ply"))
        min_bound, max_bound = np.array(ref_mesh.bounds)
        offset = (max_bound + min_bound) / 2
        scale = 1.86 / (max_bound - min_bound).max()  # within bound [-0.93, 0.93]
        norm_vertices = np.asarray(ref_mesh.vertices)
        norm_vertices -= offset
        norm_vertices *= scale
        ref_mesh.vertices = norm_vertices
        ref_mesh.export(os.path.join(work_dir, f"mvs_opencv_poisson_norm_zup_norm.ply"))

        poses_norm_zup = []
        for pose_nor in poses_norm:
            rescale_pose = np.eye(4)
            rescale_pose[:3, :4] = rxyz[:3, :3] @ pose_nor[:3, :4]
            rescale_pose[:3, 3] = (rescale_pose[:3, 3] - offset) * scale
            poses_norm_zup.append(rescale_pose)
        poses_norm_zup = np.stack(poses_norm_zup)

        # Save the pose
        pose_trans_path
        for pose, posfname in zip(poses_norm_zup, pose_origin_files):
            np.savetxt(
                os.path.join(pose_trans_path, os.path.basename(posfname)), 
                pose @ np.diag([1, -1, -1, 1])    # OpenCV -> OpenGL
            )


        ##########################################################################
        ''' Step four: render the foreground mask for the object'''
        mesh_render = o3d.io.read_triangle_mesh(
            os.path.join(work_dir, "mvs_opencv_poisson_norm_zup_norm.ply")
        )
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_render)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        pose_files = listfiles(
            os.path.join(
                work_dir, "pose"
            )
        )
        pose_files.sort(key=lambda x: int(os.path.basename(x)[:-4].split("_")[1]))
        poses = np.stack([np.loadtxt(posf) for posf in pose_files])

        image_files = listfiles(
            os.path.join(
                work_dir, "images"
            )
        )
        image_files.sort(key=lambda x: int(os.path.basename(x)[:-4].split("_")[1]))
        
        mask_path = os.path.join(work_dir, "masks")
        os.makedirs(mask_path, exist_ok=True)
        for pose, imgf in tqdm(zip(poses, image_files)):
            origins = pose[:3, 3]  # [3]
            origins = np.tile(origins, (height * width, 1))  # [H * W, 3]
            rot_mat = pose[:3, :3]
            dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

            # ray casting to get the depth&normal from reconstructed mesh
            rays = np.concatenate([origins, dirs], axis=1)  # [H * W, 6]
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays, nthreads=0)
            depth = ans["t_hit"].numpy().reshape(height, width)
            depth[np.isinf(depth)] = 0
            depth[depth > 0.05] = 1
            depth[depth <= 0.05] = 0
            np.save(os.path.join(mask_path, f"{os.path.basename(imgf)[:-4]}.npy"), depth)
            img = imageio.imread(imgf)
            img[depth<0.5] = 0
            imageio.imsave(os.path.join(mask_path, f"{os.path.basename(imgf)}"), img)
