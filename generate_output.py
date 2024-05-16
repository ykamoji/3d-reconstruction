import os
import imageio
import math
import torch
import numpy as np
from tqdm import tqdm
from renderer.utils import TensorGroup, colorize, sample_front_circle
from renderer.camera import compute_cam2world_matrix, sample_rays
from torchvision.utils import make_grid
from metrics import FrechetInceptionDistanceModified, compute_flatness_score
from torchmetrics.image.inception import InceptionScore

FID = FrechetInceptionDistanceModified(feature=64, reset_real_features=False, normalize=True)
IS = InceptionScore(feature=64, normalize=True)


def create_3d_output(model, planes, num_frames=128, vimg_size=128, output_path='results/', filename=0, gt_imgs=None):
    device = planes.device
    bs = planes.shape[0]
    camera_params = TensorGroup(
        angles=torch.zeros(1 ,3),
        fov=torch.ones(1 ) *18,
        radius=torch.ones(1 ) *5,
        look_at=torch.zeros(1 ,3),
    )
    camera_params.angles[:, 0] = camera_params.angles[:, 0 ] +np.pi /2
    camera_params.angles[:, 1] = camera_params.angles[:, 1 ] +np.pi /2

    camera_samples = sample_front_circle(camera_params, num_frames)
    cam2w = compute_cam2world_matrix(camera_samples)

    ray_origins, ray_directions = sample_rays(cam2w, camera_samples.fov[:, None], [vimg_size ,vimg_size])

    frames = []
    depths = []
    print('Visualizing file: ', filename)
    pbar = tqdm(range(num_frames))
    for th in pbar:

        rays_o, rays_d = ray_origins[th].to(device), ray_directions[th].to(device)
        rays_o, rays_d = rays_o[None].repeat(bs, 1, 1), rays_d[None].repeat(bs, 1, 1)

        rgb_out, depth, _, _ = model.unet_model.renderer(planes, model.unet_model.decoder, rays_o, rays_d,
                                                         model.unet_model.rendering_options)

        rgb = (rgb_out +1 )* 0.5
        rgb_reshape = rgb.reshape(bs, vimg_size, vimg_size, 3).permute(0 ,3 ,1 ,2).cpu()
        depth_reshape = depth.reshape(bs, vimg_size, vimg_size, 1).permute(0 ,3 ,1 ,2).cpu()
        depth_reshape = colorize(depth_reshape, cmap='magma_r')
        depth_reshape = torch.from_numpy(depth_reshape).to(rgb_reshape.device)
        if bs == 1:
            depth_reshape = depth_reshape.unsqueeze(0)
        depth_reshape = depth_reshape.permute(0 ,3 ,1 ,2)[: ,:3 ] /255
        depths.append(depth_reshape)
        combined = torch.cat([rgb_reshape, depth_reshape], dim=3)
        combined = make_grid(combined, nrow = int(math.sqrt(bs)))
        frame = (255 * np.clip(combined.permute(1 ,2 ,0).cpu().detach().numpy(), 0, 1)).astype(np.uint8)
        frames.append(frame)

        x_canon = rgb_out.permute(0, 2, 1).reshape(-1, 3, 128, 128)

        if gt_imgs != None:
            FID.update(gt_imgs.to(torch.uint8).cpu(), real=True)
            FID.update(x_canon.to(torch.uint8).cpu(), real=False)
            IS.update(x_canon.to(torch.uint8).cpu())

            if th > 2:
                pbar.set_postfix({"FID": f"{FID.compute():.8f}", "IS": IS.compute()})

    imageio.mimwrite(os.path.join(output_path, f'output-{filename}.mp4'), frames, fps=20, quality=8)

    print(f"NFS= {compute_flatness_score(depths):.5f}")

    return depths