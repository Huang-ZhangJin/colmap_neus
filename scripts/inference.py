# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import argparse
from typing import Optional, Sequence

import cumcubes
import numpy as np
import omegaconf
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

import cneus
from cneus.utils.mesh import (
    evaluate_mesh, 
    refuse, 
    transform, 
    remove_isolate_component_by_diameter,
    o3dmesh_to_trimesh
    )

os.environ["PYOPENGL_PLATFORM"] = "egl"

cneus.set_random_seed(123456)


def get_args() -> argparse.Namespace:
    parser = cneus.get_default_args()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file to load")
    parser.add_argument(
        "--marching_cube_resolution",
        "--mc_res",
        type=int,
        default=512,
        help="the resolution of marching cubes",
    )
    parser.add_argument(
        "--marching_cube_threshold",
        "--mc_thres",
        type=float,
        default=0,
        help="the density threshold of marching cubes, 0 for SDF",
    )
    parser.add_argument(
        "--mesh_clean_percentage",
        "-mcp",
        type=float,
        default=0.,
        help="percentage to clean the mesh",
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def eval_mesh(
    neus: cneus.NeuS,
    dataset: cneus.DatasetBase,
    conf: omegaconf.OmegaConf,
    epoch_id: int = 0,
    iter_id: int = 0,
    save_dir: Optional[str] = None,
    scale: float = 1.0,
    offset: Sequence[float] = [0, 0, 0],
    mesh_clean_percent: float = 0.,
):
    resx = int(conf.marching_cube_resolution)
    resy = int(conf.marching_cube_resolution)
    resz = int(conf.marching_cube_resolution)

    bound = conf.bound
    half_grid_size = bound / conf.marching_cube_resolution
    xs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resx)
    ys = np.linspace(-bound + half_grid_size, bound - half_grid_size, resy)
    zs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    samplepos = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
    samplepos = torch.from_numpy(samplepos).float().cuda()

    batch_size = 720720
    all_sdfgrid = torch.Tensor([]).cuda()
    for i in tqdm(range(0, len(samplepos), batch_size)):
        with torch.no_grad():
            sample_vals_sdf = neus.sdf_net.sdf(samplepos[i : i + batch_size])
        all_sdfgrid = torch.cat([all_sdfgrid, sample_vals_sdf])
        del sample_vals_sdf
        torch.cuda.empty_cache()

    sdfgrid = all_sdfgrid.view(resx, resy, resz)
    sdfgrid = sdfgrid.reshape(resx, resy, resz)

    vertices, faces = cumcubes.marching_cubes(
        sdfgrid,
        float(conf.marching_cube_threshold),
        ([-bound] * 3, [bound] * 3),
        verbose=False,
    )

    tri_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())

    pred_mesh = refuse(tri_mesh, dataset, scale)
    if save_dir:
        tri_mesh.export(os.path.join(save_dir, f"predicted_raw_ep{epoch_id}_iter{iter_id}.ply"))
        o3d.io.write_triangle_mesh(
            os.path.join(save_dir, f"predicted_n_ep{epoch_id}_iter{iter_id}.ply"), pred_mesh
        )

    pred_mesh = transform(pred_mesh, scale, offset)
    if save_dir:
        o3d.io.write_triangle_mesh(os.path.join(save_dir, f"predicted_ep{epoch_id}_iter{iter_id}.ply"), pred_mesh)

    mesh_gt = o3d.io.read_triangle_mesh(
        # os.path.join(dataset.data_root, dataset.scene, f"{dataset.scene}_rotgt_clean.ply")
        os.path.join(dataset.data_root, dataset.scene, f"{dataset.scene}_manhattansdf.obj")
    )
    evaluate_result = evaluate_mesh(pred_mesh, mesh_gt)
    for k, v in evaluate_result.items():
        print(f"{k:7s}: {v:1.3f}")

    if mesh_clean_percent>0.001:
        # Eval after cleaning
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        o3d_mesh = remove_isolate_component_by_diameter(o3d_mesh, mesh_clean_percent)

        tri_mesh = o3dmesh_to_trimesh(o3d_mesh)
        pred_mesh = refuse(tri_mesh, dataset, scale)
        if save_dir:
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"predicted_raw_ep{epoch_id}_cleaned.ply"), o3d_mesh
            )
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"predicted_n_ep{epoch_id}_cleaned.ply"), pred_mesh
            )

        pred_mesh = transform(pred_mesh, scale, offset)
        if save_dir:
            o3d.io.write_triangle_mesh(os.path.join(save_dir, f"predicted_ep{epoch_id}_cleaned.ply"), pred_mesh)

        evaluate_result = evaluate_mesh(pred_mesh, mesh_gt)
        print(f"Score after cleaning the predicted mesh with diameter percentage {mesh_clean_percent} : ")
        for k, v in evaluate_result.items():
            print(f"{k:7s}: {v:1.3f}")


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = cneus.merge_config_file(args)
    print("Config:\n", omegaconf.OmegaConf.to_yaml(conf))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # update the model parameters
    conf.model.bound = conf.bound
    conf.model.grid_res = conf.grid_res
    neus = cneus.NeuS(**conf.model).to(device)
    neus.eval()
    checkpoint = cneus.load_checkpoint(neus, conf.ckpt)
    epoch_id = int(checkpoint["meta"]["epoch_id"])
    iter_id = int(checkpoint["meta"]["gstep_id_base"])
    # iter_id = int(conf.ckpt[:-4].split("_")[-1])

    # # load normalize matrix to get the origin size
    # trans_file = os.path.join(conf.data_dir, conf.scene, "trans_n2w.txt")
    # if os.path.exists(trans_file):
    #     trans_n2w = np.loadtxt(trans_file)
    #     scale = trans_n2w[0, 0]
    #     offset = trans_n2w[:3, 3]
    # else:
    scale = 1.0
    offset = np.array([0., 0., 0.])
    print(f"mesh scale {scale} | mesh offsets {offset}")

    print("begin marching cubes to export mesh")
    neus.export_mesh(
        filename=conf.ckpt.replace(".pth", ".ply"),
        resolution=conf.marching_cube_resolution,
        batch_size=720720,
    )
    norm_mesh = o3d.io.read_triangle_mesh(conf.ckpt.replace(".pth", ".ply"))
    if float(conf.mesh_clean_percentage)>0.001:
        clean_mesh = remove_isolate_component_by_diameter(
            norm_mesh,
            float(conf.mesh_clean_percentage),
            keep_mesh=False
        )
        o3d.io.write_triangle_mesh(conf.ckpt.replace(".pth", "_clean.ply"), clean_mesh)
    print("Finished marching cube")
    

