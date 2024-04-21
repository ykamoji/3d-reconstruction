import torch
import yaml
import os
import json
import imageio
import math
import numpy as np
from tqdm import tqdm
from renderer.utils import TensorGroup, colorize, sample_front_circle, get_yaml_loader, CustomObject
from renderer.camera import compute_cam2world_matrix, sample_rays
from renderer.g3dr_renderer import ImportanceRenderer_extended
from eg3d.renderer import ImportanceRenderer
from renderer.model import OSGDecoder_extended, Unet
from generator_model import initialize_model
from dataset import getDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def generate_video(model, planes, num_frames=128, vimg_size=128, output_path='results/', filename=0):
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
    print('Visualizing file: ', filename)
    for th in tqdm(range(num_frames)):

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

        combined = torch.cat([rgb_reshape, depth_reshape], dim=3)
        combined = make_grid(combined, nrow = int(math.sqrt(bs)))
        frame = (255 * np.clip(combined.permute(1 ,2 ,0).cpu().detach().numpy(), 0, 1)).astype(np.uint8)
        frames.append(frame)

    imageio.mimwrite(os.path.join(output_path, f'output-{filename}.mp4'), frames, fps=40, quality=8)


def render_image(config):

    render_3d = config.rendering.render
    unet_feature_dim = config.unet_feature_dim
    results_folder = config.logging.save_dir

    if render_3d:
        if config.rendering.extended_renderer:
            # print('Extending EG3D')
            renderer = ImportanceRenderer_extended()
        else:
            renderer = ImportanceRenderer()

        eg3d_decoder = OSGDecoder_extended(unet_feature_dim,
                                           options=config.rendering.triplane_renderer_config.mlp_decoder_config)
        unet_out_dim = config.unet_feature_dim * 3
    else:
        renderer = None
        eg3d_decoder = None
        unet_out_dim = 3

    device = torch.device(config.device)

    unet_model = Unet(
        channels=4,
        dim=config.dim,
        out_dim=unet_out_dim,
        renderer=renderer,
        eg3d_decoder=eg3d_decoder,
        rendering_options=config.rendering.triplane_renderer_config.rendering_kwargs,
        dim_mults=(1, 2, 4, 8),
        render_3d=render_3d,
        image_size=config.image_size,
        config=config
    )

    generator3DModel = initialize_model(config, unet_model)

    model = generator3DModel()
    model.to(device)

    checkpoint = torch.load(f'{config.logging.model_path}', map_location=config.device)

    if 'pytorch-lightning_version' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        state_dict = {'unet_model.' + k.partition('module.')[2]: checkpoint['model'][k] for k in
                      checkpoint['model'].keys()}

    model.load_state_dict(state_dict)

    datasetClass = getDataset(config.training.dataset)
    dataset = datasetClass(config.visualization.image_path, image_size=config.image_size, config=config)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=True,
                            num_workers=8, persistent_workers=True)

    model.eval()
    step=0
    for data in iter(dataloader):
        with torch.no_grad():
            step += 1
            # bs = data['images'].shape[0]
            x_start = 2 * data['images'] - 1
            depth_start = data['depth']
            input_feat = torch.cat([x_start, depth_start], dim=1).to(device)
            _, _, _, planes = model(input_feat, return_3d_features=True, render=False)
            generate_video(model, planes, output_path=config.logging.save_dir, filename=step)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    config = json.loads(x, object_hook=lambda d: CustomObject(**d))

    render_image(config)
