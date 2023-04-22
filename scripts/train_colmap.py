# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import math
import os

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cneus


cneus.set_random_seed(123456)
os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_args() -> argparse.Namespace:
    parser = cneus.get_default_args()

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="total number of iters to optimize for learning rate scheduler",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=30000,
        help="total number of iters to optimize for learning rate scheduler",
    )
    parser.add_argument("--batch_size", type=int, default=5000, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate or mlp")
    parser.add_argument("--ek_lambda", type=float, default=0.0, help="weight for eikonal loss")

    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training images. Defaults to use all avaiable.",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=None,
        help="Number of evaluation images. Defaults to use all avaiable.",
    )
    parser.add_argument("--print_every", type=int, default=10, help="print every")
    parser.add_argument("--save_every", type=int, default=1, help="save every x epochs")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "l2", "l1"],
        help="loss type",
    )
    parser.add_argument(
        "--up_sample_steps",
        type=int,
        default=0,
        help="Number of up sampling. Defaults to 0",
    )

    parser.add_argument("--export_mesh", type=str, default="vis_mc/sdf_mc.ply")

    parser.add_argument("--load_ckpt", type=str, default=None)

    parser.add_argument("--vis_normal", action="store_true", default=False)

    # logging
    parser.add_argument("--log_image", action="store_true", default=False)
    parser.add_argument("--log_depth", action="store_true", default=False)

    args = parser.parse_args()
    return args


def train_epoch(
    neus: cneus.NeuS,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    opt: cneus.RenderOptions,
    conf: omegaconf.OmegaConf,
    dataset_train: cneus.ColmapDataset,
    gstep_id_base: int,
    update_grid_sam_res: bool,
    epoch_id: int,
) -> int:
    neus.update_occupancy_grid(opt=opt, update_grid_sam_res=update_grid_sam_res)
    neus.export_mesh(filename=conf.export_mesh)

    print("Train epoch:")
    num_rays = dataset_train.rays.origins.size(0)
    num_iters_per_epoch = (num_rays - 1) // conf.batch_size + 1
    pbar = tqdm(enumerate(range(0, num_rays, conf.batch_size)), total=num_iters_per_epoch)
    stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}

    criternion: nn.Module
    if conf.loss_type == "smoothl1":
        criternion = nn.SmoothL1Loss(beta=0.1)
    elif conf.loss_type == "l2":
        criternion = nn.MSELoss()
    elif conf.loss_type == "l1":  # does not work
        criternion = nn.L1Loss()
    depth_criternion = nn.L1Loss(reduction="none")

    neus.export_mesh(filename=conf.export_mesh)

    for iter_id, batch_begin in pbar:
        gstep_id = iter_id + gstep_id_base

        update_grid_freq = 16
        if (gstep_id + 1) % update_grid_freq == 0:
            neus.update_occupancy_grid(opt=opt)

        batch_end = min(batch_begin + conf.batch_size, num_rays)
        batch_origins = dataset_train.rays.origins[batch_begin:batch_end]
        batch_dirs = dataset_train.rays.dirs[batch_begin:batch_end]
        rgb_gt = dataset_train.rays.gt[batch_begin:batch_end]
        ray_mask = dataset_train.rays.mask[batch_begin:batch_end]
        rays = cneus.Rays(batch_origins, batch_dirs)

        export_mesh_freq = 1000
        if (gstep_id + 1) % export_mesh_freq == 0:
            neus.export_mesh(
                filename=str(conf.export_mesh).replace(".ply", f"_iter_{gstep_id}.ply"),
                resolution=256 if epoch_id < 90 else 512,
                batch_size=64**3,
            )

        # cos_anneal = min(1.0, (gstep_id + 1) / (50000 * 512 / conf.batch_size))
        cos_anneal = 1.0
        optimizer.zero_grad()

        up_sample_steps = conf.up_sample_steps if gstep_id >= 10000 else 0
        (rgb_pred, depth_pred, normal_pred, normals, sval, _, weight_sum) = neus.train_render_outside_cuda(
            rays,
            opt,
            ek_bound = 1.2,
            cos_anneal_ratio=cos_anneal,
            up_sample_steps=up_sample_steps,
        )

        ######################################
        """ RGB loss """
        masked_rgb_pred = rgb_pred[ray_mask]
        masked_rgb_gt = rgb_gt[ray_mask]
        rgb_loss = criternion(masked_rgb_pred, masked_rgb_gt)
        """ Eikonal loss """
        ek_loss = float(conf.ek_lambda) * normals
        loss = rgb_loss + ek_loss
        """ Ray mask loss """
        ray_mask_loss = F.binary_cross_entropy(
            weight_sum.squeeze().clip(1e-3, 1.0 - 1e-3), 
            ray_mask.float()
        )
        loss += 0.1 * ray_mask_loss

        ######################################
        loss.backward()
        optimizer.step()
        scheduler.step()

        rgb_pred.clamp_max_(1.0)
        mse = F.mse_loss(rgb_pred[ray_mask], rgb_gt[ray_mask])

        # Stats
        mse_num: float = mse.detach().item()
        psnr = -10.0 * math.log10(mse_num)
        stats["mse"] += mse_num
        stats["psnr"] += psnr
        stats["invsqr_mse"] += 1.0 / mse_num**2

        if (iter_id + 1) % conf.print_every == 0:
            # Print averaged stats
            pbar.set_description(
                f"epoch {epoch_id} iter {gstep_id} psnr={psnr:.2f} "
                f"ek={ek_loss:.4f} mask={ray_mask_loss:.4f} "
            )
            for stat_name in stats:
                stat_val = stats[stat_name] / conf.print_every
                summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                stats[stat_name] = 0.0
            summary_writer.add_scalar(
                "Params/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_step=gstep_id,
            )
            summary_writer.add_scalar("Params/cos_anneal_ratio", cos_anneal, global_step=gstep_id)
            summary_writer.add_scalar("Params/s_val", sval, global_step=gstep_id)
            # Loss
            summary_writer.add_scalar("Loss/total", loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/rgb", rgb_loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/ek", ek_loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/mask", ray_mask_loss, global_step=gstep_id)

    gstep_id_base += len(pbar)

    return gstep_id_base


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = cneus.merge_config_file(args)
    print("Config:\n", omegaconf.OmegaConf.to_yaml(conf))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init train directory summary writer
    os.makedirs(conf.train_dir, exist_ok=True)
    summary_writer = SummaryWriter(conf.train_dir)

    conf.model.bound = conf.bound
    conf.model.grid_res = conf.grid_res
    print(f"Using Occupancy Grids (bound : {conf.bound} | res : {conf.grid_res})")
    neus = cneus.NeuS(**conf.model).to(device)
    neus.train()
    optimizer = optim.Adam(neus.parameters(), lr=conf.lr)
    scheduler = cneus.NeusScheduler(optimizer, warm_up_end=5000, total_steps=conf.n_iters*conf.n_epochs)

    # set render options
    opt = cneus.RenderOptions()
    cneus.setup_render_opts(opt, conf)
    print("Render options:\n", opt)

    # init dataset
    factor = 1
    train_dataset = cneus.datasets[conf.dataset_type](
        conf.data_dir,
        split="train",
        device=device,
        factor=factor,
        patch_size=1,
        n_images=conf.n_train,
    )

    # prepare for training
    ckpt_path = os.path.join(conf.train_dir, "ckpt.pth")
    batch_size = conf.batch_size

    num_epochs = args.n_epochs
    gstep_id_base = 0
    for epoch_id in range(num_epochs):
        # wheter to reload from a checkpoint
        if conf.load_ckpt is not None:
            ckpt_dict = cneus.load_checkpoint(neus, str(conf.load_ckpt))
            print(f"loadckpt from {conf.load_ckpt}")
            epoch_id = int(ckpt_dict["meta"]["epoch_id"])
            gstep_id_base = int(ckpt_dict["meta"]["gstep_id_base"])

        # whether upsample the grid sampling resolution
        update_grid_sam_res = (epoch_id + 1) % conf.ups_epoch == 0 and epoch_id > 0

        # training
        train_dataset.shuffle_rays(
            epoch_size=conf.batch_size * conf.n_iters
        )

        gstep_id_base = train_epoch(
            neus,
            optimizer,
            scheduler,
            opt,
            conf,
            train_dataset,
            gstep_id_base,
            update_grid_sam_res,
            epoch_id,
        )

        # save
        if conf.save_every > 0 and (epoch_id + 1) % max(factor, conf.save_every) == 0:
            print("Saving", ckpt_path)
            cneus.save_checkpoint(
                neus,
                ckpt_path,
                optimizer,
                scheduler,
                dict(epoch_id=epoch_id + 1, gstep_id_base=gstep_id_base + 1),
            )
            cneus.save_checkpoint(
                neus,
                ckpt_path.replace(".pth", f"_{epoch_id + 1}.pth"),
                optimizer,
                scheduler,
                dict(epoch_id=epoch_id + 1, gstep_id_base=gstep_id_base + 1),
            )