import yaml
import json
import os
import clip
import torch
from torch.optim import Adam
import lightning as L
from lightning import Trainer
from shutil import rmtree
from pathlib import Path
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from utils import CustomObject, get_yaml_loader
from dataset import Imagenet_Dataset
from g3dr_renderer import ImportanceRenderer_extended
from eg3d.renderer import ImportanceRenderer
from model import OSGDecoder_extended, Unet

STYLE = 'lod_no'
GRAD_CLIP_MIN = 0.05
w_weight, w_depth, w_clip, w_tv, w_perceptual, w_rgb = 0.0, 2, 0.35, 0.1, 2, 1


class Generator(L.LightningModule):

    def __init__(self, model):
        super(Generator, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, log_name="train", on_epoch=False, on_step=True)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, log_name="val", on_epoch=True, on_step=False)

    def _common_step(self, batch, batch_idx, log_name="", on_epoch=True, on_step=True):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        if config.training.dataset == 'Imagenet':
            dataset = Imagenet_Dataset(config.training.dataset_folder, image_size=image_size, config=config)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        pass


def train(config):

    dim = 128
    image_size = 128


    rendering_kwargs = config.rendering.triplane_renderer_config.rendering_kwargs
    mlp_decoder_config = config.rendering.triplane_renderer_config.mlp_decoder_config
    bs = config.training.batch_size
    learning_rate = float(config.training.learning_rate)
    save_and_sample_every = config.training.save_and_sample_every
    results_folder = config.logging.save_dir
    version = config.logging.version
    render_3d = config.rendering.render

    unet_feature_dim = config.unet_feature_dim

    version = version + f'_{STYLE}_grad{GRAD_CLIP_MIN:.2f}_{image_size:.0f}_wd{w_depth:.2f}_ww{w_weight:.4f}_wp{w_perceptual:.2f}_wc{w_clip:.2f}_w_tv{w_tv:.4f}_lr{learning_rate:.6f}'
    print('Version:: ', version)

    results_folder_path = Path(results_folder)
    results_folder_path.mkdir(exist_ok=True)
    save_dir = results_folder + '/{}/{}'
    save_dir_images = save_dir.format(version, 'images')
    save_dir_checkpoints = save_dir.format(version, 'checkpoints')
    save_dir_tb = save_dir.format(version, 'runs')
    create_new = config.logging.create_new
    load_model = config.logging.load_model
    resume = False

    if not resume:
        if os.path.exists(save_dir.format(version, '')):
            rmtree(save_dir.format(version, ''))
        if create_new:
            os.makedirs(save_dir.format(version, ''))
            os.makedirs(save_dir_checkpoints)
            os.makedirs(save_dir_images)
            os.makedirs(save_dir_tb)
    if render_3d:
        if config.rendering.extended_renderer:
            # print('Extending EG3D')
            renderer = ImportanceRenderer_extended()
        else:
            renderer = ImportanceRenderer()

        eg3d_decoder = OSGDecoder_extended(unet_feature_dim, options=mlp_decoder_config)
        unet_out_dim = config.unet_feature_dim * 3
    else:
        renderer = None
        eg3d_decoder = None
        unet_out_dim = 3

    train_num_steps = config.training.train_num_steps

    model = Unet(
        channels=4,
        dim=dim,
        out_dim=unet_out_dim,
        renderer=renderer,
        eg3d_decoder=eg3d_decoder,
        rendering_options=rendering_kwargs,
        dim_mults=(1, 2, 4, 8),
        render_3d=render_3d,
        image_size=image_size,
        config=config
    )

    clip_model, _ = clip.load("ViT-B/16", device='mps', download_root='model/')
    znormalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if load_model:
       pass



if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    config = json.loads(x, object_hook=lambda d: CustomObject(**d))

    train(config)