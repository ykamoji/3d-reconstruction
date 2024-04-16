import yaml
import json
import os
import clip
import torch
import numpy as np
from camera import compute_cam2world_matrix, sample_rays
from random import random
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins import MixedPrecision
from shutil import rmtree
from pathlib import Path
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from utils import TensorGroup, CustomObject, get_yaml_loader
from dataset import getDataset, get_camera_rays
from g3dr_renderer import ImportanceRenderer_extended
from eg3d.renderer import ImportanceRenderer
from model import OSGDecoder_extended, Unet
import lpips

STYLE = 'lod_no'
GRAD_CLIP_MIN = 0.05
w_weight, w_depth, w_clip, w_tv, w_perceptual, w_rgb = 0.0, 2, 0.35, 0.1, 2, 1


def linear_high2low(step, start_value, final_value, start_iter, end_iter):
    return max(final_value, start_value - ((start_value - final_value)*(step-start_iter)/(end_iter - start_iter)))


def linear_low2high( step, start_value, final_value, start_iter, end_iter):
    return min(final_value, start_value + ((final_value - start_value) * (step - start_iter) / (end_iter - start_iter)))


def tv_loss(img):
    w_variance = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = (h_variance + w_variance)
    return loss


def unnorm(t):
    return (t + 1) * 0.5


class Generator(L.LightningModule):

    def __init__(self, model, clip_model, perceptual_criterion, config):
        super(Generator, self).__init__()
        self.model = model
        self.device_override = torch.device(config.device)
        self.config = config
        self.znormalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_model = clip_model
        self.perceptual_criterion = perceptual_criterion
        self.loss_fn = F.l1_loss

    def forward(self, x, **kwargs):
        self.model.to(self.device_override)
        return self.model(x, **kwargs)

    def training_step(self, data, batch_idx):
        data_images = data['images'].to(self.device_override)
        data_depth = data['depth'].to(self.device_override)

        batch_size = data_images.shape[0]
        x_start = (2 * data_images - 1)
        depth_start = data_depth
        input_feat = torch.cat([x_start, depth_start], dim=1)

        if random() > min(0.4, 2 * self.global_step / self.config.training.train_num_steps):
            Process = self.process_one
        else:
            Process = self.process_two

        pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip, \
            multiply_w_perceptual, loss_tv, w1 = Process(input_feat, data_images, data_depth, x_start, batch_size)

        loss_perceptual = loss_perceptual.mean()
        loss_rgb = self.loss_fn(pred_imgs, gt_imgs).mean()
        loss_depth = self.loss_fn(pred_depth, gt_depth, reduction='mean')
        loss_clip = self.loss_fn(pred_clip, gt_clip, reduction='mean')
        loss_weight = (1 - w1).mean()
        multiply_weight = linear_low2high(self.global_step, 0, 1, 0, self.config.training.train_num_steps // 2)

        loss_total = loss_rgb * w_rgb + loss_clip * multiply_w_clip + w_depth * loss_depth + loss_perceptual * multiply_w_perceptual + loss_tv * w_tv + loss_weight * multiply_weight

        for name, loss in [ ("rgb", loss_rgb), ("perceptual", loss_perceptual), ("depth", loss_depth),
                            ("clip", loss_clip), ("tv", loss_tv) , ("total", loss_total)]:
            self.log_loss(name+"_loss", loss)

        return loss_total.to(self.device_override)

    def process_one(self, input_feat, data_images, data_depth, x_start, batch_size):
        _, _, _, planes_old = self(input_feat, return_3d_features=True, render=False)
        rays_o , rays_d = get_camera_rays(config.image_size)
        rays_o = rays_o[None].repeat(batch_size, 1, 1).to(self.device_override)
        rays_d = rays_d[None].repeat(batch_size, 1, 1).to(self.device_override)

        x_canon, depth, w1, depth_ref = self.model.renderer(planes_old, self.model.decoder,
                                                       rays_o, rays_d,
                                                       self.model.rendering_options, importance_depth=data_depth)

        x_canon = x_canon.permute(0, 2, 1).reshape(-1, 3, config.image_size, config.image_size)
        # depth_ref = depth_ref[:, :, :, 0].permute(0, 2, 1).reshape(batch_size, -1, config.image_size, config.image_size)

        depth = depth.permute(0, 2, 1).reshape(-1, 1, config.image_size, config.image_size)


        if config.image_size < 224:
            gt_input_img =  F.interpolate(data_images, size=[224, 224], mode='bilinear')
            x_canon_resize =  F.interpolate(x_canon, size=[224, 224], mode='bilinear')
        else:
            gt_input_img = data_images
            x_canon_resize = x_canon

        gt_normalized = self.znormalize(gt_input_img)
        clip_embed_gt = self.clip_model.to(self.device_override).encode_image(gt_normalized)
        x_canon_normalized = self.znormalize(unnorm(x_canon_resize))
        clip_embed_x_canon = self.clip_model.encode_image(x_canon_normalized)

        pred_imgs, pred_depth, gt_imgs, gt_depth = x_canon, depth, x_start, data_depth
        pred_clip, gt_clip = clip_embed_gt, clip_embed_x_canon
        # 128 only
        loss_perceptual = self.perceptual_criterion.to(self.device_override)(x_start, x_canon)
        if self.global_step > self.config.training.train_num_steps // 2:
            multiply_w_clip = w_clip / 2
            multiply_w_perceptual = w_perceptual / 2
        else:
            multiply_w_clip = 0
            multiply_w_perceptual = 0

        return pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip,\
                multiply_w_perceptual, 0, w1

    def process_two(self, input_feat , data_images, data_depth , x_start, batch_size):
        fov_degrees = torch.ones(batch_size, 1).to(self.device_override) * self.config.rendering.camera_config.fov
        zero = torch.tensor([0, ], dtype=torch.float)
        _, _, _, planes_old = self(input_feat, return_3d_features=True, render=False)

        camera_params = TensorGroup(
            angles=torch.zeros(batch_size, 3),
            radius=torch.ones(batch_size) * self.config.rendering.camera_config.radius,
            look_at=torch.zeros(batch_size, 3),
        )

        start_diff = 24
        final_diff = 6
        start_iter = 0
        end_iter = self.config.training.train_num_steps // 3
        denominator = linear_high2low(self.global_step, start_diff, final_diff, start_iter, end_iter)

        start_clip = w_clip / 16
        # multiply_w_clip = min(w_clip, start_clip + ((w_clip - start_clip) * ((self.global_step - start_iter) / (end_iter - start_iter))))

        sample_diff1 = (torch.rand(batch_size) * 2 - 1) * np.pi / denominator
        sample_diff2 = (torch.rand(batch_size) * 2 - 1) * np.pi / 18

        camera_params.angles[:, 0] = np.pi / 2 + sample_diff1
        camera_params.angles[:, 1] = np.pi / 2 + sample_diff2

        cam2w = compute_cam2world_matrix(camera_params).to(self.device_override)
        ray_origins, ray_directions = sample_rays(cam2w, fov_degrees, (self.config.image_size, self.config.image_size))

        x_novel, depth_novel, w1, depth_ref = self.model.renderer(planes_old, self.model.decoder,
                                                             ray_origins, ray_directions,
                                                             self.model.rendering_options)
        depth_novel = depth_novel.permute(0, 2, 1).reshape(-1, 1, self.config.image_size, self.config.image_size)
        x_novel_128 = x_novel.permute(0, 2, 1).reshape(-1, 3, self.config.image_size, self.config.image_size)

        # resize to 224 if different, so that it works with clip
        if self.config.image_size < 224:
            gt_input_img = torch.nn.functional.interpolate(data_images, size=[224, 224], mode='bilinear')
            x_novel = torch.nn.functional.interpolate(x_novel_128, size=[224, 224], mode='bilinear')
        else:
            gt_input_img = data_images

        gt_normalized = self.znormalize(gt_input_img)
        clip_embed_gt = self.clip_model.to(self.device_override).encode_image(gt_normalized)
        novel_normalized = self.znormalize(unnorm(x_novel))
        clip_embed_trans = self.clip_model.encode_image(novel_normalized)

        pred_imgs, pred_depth, gt_imgs, gt_depth = zero, zero, zero, zero
        pred_clip, gt_clip = clip_embed_trans, clip_embed_gt

        loss_tv = tv_loss(depth_novel)

        loss_perceptual_log, loss_perceptual_all = \
            self.perceptual_criterion.to(self.device_override)(x_start, x_novel_128, retPerLayer=True)
        loss_perceptual = 0
        weights_perceptual = [0, 0, 0, 1, 1]
        for i in range(5):
            loss_perceptual += loss_perceptual_all[i].mean() * weights_perceptual[i]

        multiply_w_perceptual = w_perceptual
        multiply_w_clip = w_clip

        return pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip, \
            multiply_w_perceptual, loss_tv, w1

    def log_loss(self, name, loss):
        self.log(f"{name}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.config.training.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=float(self.config.training.learning_rate), betas=(0.9, 0.99))
        scheduler = OneCycleLR(optimizer, max_lr=float(self.config.training.learning_rate),
                               total_steps=(self.config.training.train_num_steps - self.global_step),
                               pct_start=0.02, div_factor=25)
        return ([optimizer], [scheduler])

    def train_dataloader(self):
        datasetClass = getDataset(config.training.dataset)
        dataset = datasetClass(config.training.dataset_folder, image_size=config.image_size, config=config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, drop_last=True,
                                    num_workers=8, persistent_workers=True)
        return dataloader

def train(config):

    rendering_kwargs = config.rendering.triplane_renderer_config.rendering_kwargs
    mlp_decoder_config = config.rendering.triplane_renderer_config.mlp_decoder_config
    results_folder = config.logging.save_dir
    version = config.logging.version
    render_3d = config.rendering.render
    unet_feature_dim = config.unet_feature_dim
    # version = version + f'_{STYLE}_grad{GRAD_CLIP_MIN:.2f}_\{config.image_size:.0f}_wd{w_depth:.2f}_ww{w_weight:.4f}_wp{w_perceptual:.2f}_wc{w_clip:.2f}_w_tv{w_tv:.4f}_lr{learning_rate:.6f}'
    print('Version:: ', version)

    results_folder_path = Path(results_folder)
    results_folder_path.mkdir(exist_ok=True)
    save_dir = results_folder + '/{}/{}'
    save_dir_images = save_dir.format(version, 'images')
    save_dir_checkpoints = save_dir.format(version, 'checkpoints')
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

    model_unet = Unet(
        channels=4,
        dim=config.dim,
        out_dim=unet_out_dim,
        renderer=renderer,
        eg3d_decoder=eg3d_decoder,
        rendering_options=rendering_kwargs,
        dim_mults=(1, 2, 4, 8),
        render_3d=render_3d,
        image_size=config.image_size,
        config=config
    )

    if load_model:
       pass

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir_checkpoints,
        save_top_k=1,
        verbose=True,
        monitor='total_loss_step',
        every_n_train_steps=10,
        mode='min',
        filename='imagnet-{epoch:02d}-{total_loss_step:.2f}'
    )

    trainer = Trainer(max_epochs=config.training.train_num_steps,
                      log_every_n_steps=1,
                      # precision="16-mixed",
                      default_root_dir=results_folder,
                      callbacks=[checkpoint_callback])

    clip_model, _ = clip.load("ViT-B/16", device='mps', download_root='model/')
    perceptual_criterion = lpips.LPIPS(net='vgg')
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    model = Generator(model_unet, clip_model, perceptual_criterion, config)

    trainer.fit(model)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    config = json.loads(x, object_hook=lambda d: CustomObject(**d))

    train(config)