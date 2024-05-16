import torch
import yaml
import json
from renderer.utils import get_yaml_loader, CustomObject
from renderer.g3dr_renderer import ImportanceRenderer_extended
from eg3d.renderer import ImportanceRenderer
from renderer.model import OSGDecoder_extended, Unet
from generator_model import initialize_model
from dataset import getDataset
from generate_output import create_3d_output
from torch.utils.data import DataLoader


def render_image(config):

    render_3d = config.rendering.render
    unet_feature_dim = config.unet_feature_dim

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
    dataset = datasetClass(config.visualization.image_path, image_size=config.image_size, config=config,
                           random_flip=False)
    dataloader = DataLoader(dataset, batch_size=config.training.evaluation_batch_size, num_workers=8,
                            persistent_workers=True)

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
            create_3d_output(model, planes, output_path=config.logging.save_dir, filename=step, gt_imgs=x_start)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    config = json.loads(x, object_hook=lambda d: CustomObject(**d))

    render_image(config)
