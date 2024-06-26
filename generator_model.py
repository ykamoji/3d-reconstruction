import torch
import clip
import lpips
import math
import lightning as L
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from renderer.camera import compute_cam2world_matrix, sample_rays
from renderer.utils import TensorGroup
from generate_output import create_3d_output
from random import random
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from dataset import getDataset, get_camera_rays
from torchvision.transforms import Normalize
from torchvision import utils
from metrics import FrechetInceptionDistanceModified, compute_flatness_score
from torchmetrics.image.inception import InceptionScore

_ = torch.manual_seed(123)


def initialize_model(config, unet_model):
    w_weight, w_depth, w_clip, w_tv, w_perceptual, w_rgb = 0.0, 2, 0.35, 0.1, 2, 1
    znormalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_model, _ = clip.load("ViT-B/16", download_root='model/')
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    perceptual_criterion = lpips.LPIPS(net='vgg', eval_mode=True)
    perceptual_criterion.eval()
    for param in perceptual_criterion.parameters():
        param.requires_grad = False

    FID = FrechetInceptionDistanceModified(feature=64, reset_real_features=False, normalize=True)
    IS = InceptionScore(feature=64, normalize=True)

    device = torch.device(config.device)

    clip_model.to(device)
    perceptual_criterion.to(device)

    # FID.to(device)
    # IS.to(device)

    def linear_high2low(step, start_value, final_value, start_iter, end_iter):
        return max(final_value,
                   start_value - ((start_value - final_value) * (step - start_iter) / (end_iter - start_iter)))

    def linear_low2high(step, start_value, final_value, start_iter, end_iter):
        return min(final_value,
                   start_value + ((final_value - start_value) * (step - start_iter) / (end_iter - start_iter)))

    def tv_loss(img):
        w_variance = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = (h_variance + w_variance)
        return loss

    def loss_prepare(loss_type, loss_fn, imgs_1, imgs_2):

        if loss_type == 'l1' or loss_type == 'l2' or loss_type == 'l1_smooth':
            return loss_fn(imgs_1, imgs_2)

        if loss_type == 'cos':
            ones = torch.ones(size=(imgs_1.shape[0],))
            if imgs_1.dim() < 2:
                imgs_1 = imgs_1.unsqueeze(1)
            if imgs_2.dim() < 2:
                imgs_2 = imgs_2.unsqueeze(1)

            return loss_fn(input1=torch.flatten(imgs_1, start_dim=1, end_dim=-1),
                           input2=torch.flatten(imgs_2, start_dim=1, end_dim=-1),
                           target=ones.to(imgs_1.device))

    def unnorm(t):
        return (t + 1) * 0.5

    class Generator3D(L.LightningModule):

        def __init__(self):
            super(Generator3D, self).__init__()
            self.device_override = device
            self.unet_model = unet_model
            self.config = config
            # self.loss_type = self.config.training.loss_type
            # if self.loss_type == 'l1':
            #     self.loss_fn = F.l1_loss
            # elif self.loss_type == 'l2':
            #     self.loss_fn = nn.MSELoss()
            # elif self.loss_type == 'cos':
            #     self.loss_fn = nn.CosineEmbeddingLoss()
            self.rgb_loss_fn = F.smooth_l1_loss
            self.depth_loss_fn = F.smooth_l1_loss
            self.clip_loss_fn = nn.CosineEmbeddingLoss()

        def forward(self, x, **kwargs):
            return self.unet_model(x, **kwargs)

        # def on_fit_start(self):
        #     self.trainer.fit_loop.epoch_progress.current.processed = 20
        #     self.trainer.fit_loop.epoch_progress.current.completed = 20
        #     print("Starting...")

        def on_train_start(self):
            self.unet_model.to(self.device_override).train()

        def on_train_epoch_start(self):
            self.unet_model.to(self.device_override).train()

        def on_validation_model_eval(self):
            self.unet_model.to(self.device_override).eval()

        def training_step(self, data, batch_idx):
            data_images = data['images'].to(self.device_override)
            data_depth = data['depth'].to(self.device_override)

            batch_size = data_images.shape[0]
            x_start = (2 * data_images - 1)
            depth_start = data_depth
            input_feat = torch.cat([x_start, depth_start], dim=1)

            # if input_feat.shape[1] < 4:
            #     print(f"Issue for {data['idx']}: {input_feat.shape}")
            #     print(f"{data_images.shape}")
            #     print(f"{depth_start.shape}")
            # return torch.tensor([0.0], device=self.device_override, requires_grad=True)

            if random() > min(0.4, 2 * self.global_step / self.config.training.train_num_steps):
                Process = self.process_one
            else:
                Process = self.process_two

            pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip, \
                multiply_w_perceptual, loss_tv, w1 = Process(input_feat, data_images, data_depth, x_start, batch_size)

            loss_perceptual = loss_perceptual.mean()
            loss_rgb = loss_prepare("l1_smooth", self.rgb_loss_fn, pred_imgs, gt_imgs)
            loss_depth = loss_prepare("l1_smooth", self.depth_loss_fn, pred_depth, gt_depth)
            loss_clip = loss_prepare("cos", self.clip_loss_fn, pred_clip, gt_clip)
            loss_weight = (1 - w1).mean()
            multiply_weight = linear_low2high(self.global_step, 0, 1, 0, self.config.training.train_num_steps // 2)

            loss_total = loss_rgb * w_rgb + loss_clip * multiply_w_clip + w_depth * loss_depth + loss_perceptual * multiply_w_perceptual + loss_tv * w_tv + loss_weight * multiply_weight

            for name, loss in [("rgb", loss_rgb), ("perceptual", loss_perceptual), ("depth", loss_depth),
                               ("clip", loss_clip), ("tv", loss_tv), ("total", loss_total)]:
                self.log_loss(name + "_loss", loss.item())

            return loss_total

        def process_one(self, input_feat, data_images, data_depth, x_start, batch_size):
            _, _, _, planes_old = self(input_feat, return_3d_features=True, render=False)
            rays_o, rays_d = get_camera_rays(self.config.image_size)
            rays_o = rays_o[None].repeat(batch_size, 1, 1).to(self.device_override)
            rays_d = rays_d[None].repeat(batch_size, 1, 1).to(self.device_override)

            x_canon, depth, w1, depth_ref = self.unet_model.renderer(planes_old, self.unet_model.decoder,
                                                                     rays_o, rays_d,
                                                                     self.unet_model.rendering_options,
                                                                     importance_depth=data_depth)

            x_canon = x_canon.permute(0, 2, 1).reshape(-1, 3, self.config.image_size, self.config.image_size)

            depth = depth.permute(0, 2, 1).reshape(-1, 1, self.config.image_size, self.config.image_size)

            if self.config.image_size < 224:
                gt_input_img = F.interpolate(data_images, size=[224, 224], mode='bilinear')
                x_canon_resize = F.interpolate(x_canon, size=[224, 224], mode='bilinear')
            else:
                gt_input_img = data_images
                x_canon_resize = x_canon

            gt_normalized = znormalize(gt_input_img)
            clip_embed_gt = clip_model.encode_image(gt_normalized)
            x_canon_normalized = znormalize(unnorm(x_canon_resize))
            clip_embed_x_canon = clip_model.encode_image(x_canon_normalized)

            pred_imgs, pred_depth, gt_imgs, gt_depth = x_canon, depth, x_start, data_depth
            pred_clip, gt_clip = clip_embed_gt, clip_embed_x_canon
            # 128 only
            loss_perceptual = perceptual_criterion(x_start, x_canon)
            if self.global_step > self.config.training.train_num_steps // 2:
                multiply_w_clip = w_clip / 2
                multiply_w_perceptual = w_perceptual / 2
            else:
                multiply_w_clip = 0
                multiply_w_perceptual = 0

            FID.update(gt_imgs.to(torch.uint8).cpu(), real=True)
            FID.update(pred_imgs.to(torch.uint8).cpu(), real=False)
            IS.update(pred_imgs.to(torch.uint8).cpu())

            if self.global_step > 2:
                self.log_metric("FID", FID.compute())
                self.log_metric("IS", IS.compute()[0])

            return pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip, \
                multiply_w_perceptual, torch.zeros(1, device=loss_perceptual.device), w1

        def process_two(self, input_feat, data_images, data_depth, x_start, batch_size):
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

            sample_diff1 = (torch.rand(batch_size) * 2 - 1) * np.pi / denominator
            sample_diff2 = (torch.rand(batch_size) * 2 - 1) * np.pi / 18

            camera_params.angles[:, 0] = np.pi / 2 + sample_diff1
            camera_params.angles[:, 1] = np.pi / 2 + sample_diff2

            cam2w = compute_cam2world_matrix(camera_params).to(self.device_override)
            ray_origins, ray_directions = sample_rays(cam2w, fov_degrees,
                                                      (self.config.image_size, self.config.image_size))

            x_novel, depth_novel, w1, depth_ref = self.unet_model.renderer(planes_old, self.unet_model.decoder,
                                                                           ray_origins, ray_directions,
                                                                           self.unet_model.rendering_options)
            depth_novel = depth_novel.permute(0, 2, 1).reshape(-1, 1, self.config.image_size, self.config.image_size)
            x_novel_128 = x_novel.permute(0, 2, 1).reshape(-1, 3, self.config.image_size, self.config.image_size)

            # resize to 224 if different, so that it works with clip
            if self.config.image_size < 224:
                gt_input_img = F.interpolate(data_images, size=[224, 224], mode='bilinear')
                x_novel = F.interpolate(x_novel_128, size=[224, 224], mode='bilinear')
            else:
                gt_input_img = data_images

            gt_normalized = znormalize(gt_input_img)
            clip_embed_gt = clip_model.encode_image(gt_normalized)
            novel_normalized = znormalize(unnorm(x_novel))
            clip_embed_trans = clip_model.encode_image(novel_normalized)

            pred_imgs, pred_depth, gt_imgs, gt_depth = zero, zero, zero, zero
            pred_clip, gt_clip = clip_embed_trans, clip_embed_gt

            loss_tv = tv_loss(depth_novel)

            loss_perceptual_log, loss_perceptual_all = (perceptual_criterion(x_start, x_novel_128, retPerLayer=True))
            loss_perceptual = 0
            weights_perceptual = [0, 0, 0, 1, 1]
            for i in range(5):
                loss_perceptual += loss_perceptual_all[i].mean() * weights_perceptual[i]

            multiply_w_perceptual = w_perceptual
            multiply_w_clip = w_clip

            ## Intermediate outputs for visualization
            if self.config.training.save_intermediate_images_step > 0:
                if self.global_step % self.config.training.save_intermediate_images_step == 0:
                    output_path = self.config.logging.intermediate_outputs
                    utils.save_image(data_images, str(output_path + f'/input-{self.global_step}.png'),
                                     nrow=int(math.sqrt(data_images.shape[0])))
                    utils.save_image(data_depth, str(output_path + f'/sample-{self.global_step}_depth.png'),
                                     nrow=int(math.sqrt(data_images.shape[0])), normalize=True)

                    if type(x_novel) != type(None):
                        utils.save_image(unnorm(x_novel), str(output_path + f'/sample_trans-{self.global_step}.png'),
                                         nrow=int(math.sqrt(data_images.shape[0])))
                        utils.save_image(depth_novel, str(output_path + f'/sample_trans-{self.global_step}_depth.png'),
                                         nrow=int(math.sqrt(data_images.shape[0])), normalize=True)

            return pred_imgs, gt_imgs, pred_depth, gt_depth, loss_perceptual, pred_clip, gt_clip, multiply_w_clip, \
                multiply_w_perceptual, loss_tv, w1

        def log_loss(self, name, loss):
            self.log(f"train/{name}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.config.training.batch_size)

        def log_metric(self, metric, value):
            self.log(f"eval/{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.config.training.batch_size)

        def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=float(self.config.training.learning_rate), betas=(0.9, 0.99))
            scheduler = OneCycleLR(optimizer, max_lr=float(self.config.training.learning_rate),
                                   total_steps=(self.config.training.train_num_steps - self.global_step),
                                   pct_start=0.02, div_factor=25)
            return [optimizer], [scheduler]

        def train_dataloader(self):
            datasetClass = getDataset(self.config.training.dataset)
            dataset = datasetClass(self.config.training.dataset_folder, image_size=self.config.image_size,
                                   config=self.config)
            dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, drop_last=True,
                                    num_workers=8, persistent_workers=True)
            return dataloader

        def validation_step(self, data, batch_idx):
            with torch.no_grad():
                data_images = data['images'].to(self.device_override)
                data_depth = data['depth'].to(self.device_override)
                x_start = (2 * data_images - 1)
                depth_start = data_depth
                input_feat = torch.cat([x_start, depth_start], dim=1)
                # print(f"{data['idx']}: {input_feat.shape}")
                _, _, _, planes = self(input_feat, return_3d_features=True, render=False)
                depths = create_3d_output(self, planes, output_path=config.logging.intermediate_outputs,
                                          filename=str(self.global_step) + '-' + str(batch_idx+1))

                self.log_metric("NFS", compute_flatness_score(depths))

        def val_dataloader(self):
            datasetClass = getDataset(self.config.training.dataset)
            dataset = datasetClass(self.config.training.evaluate_folder, image_size=self.config.image_size,
                                   config=self.config, random_flip=False)
            dataloader = DataLoader(dataset, batch_size=self.config.training.evaluation_batch_size, drop_last=True,
                                    num_workers=8, persistent_workers=True)
            return dataloader

    return Generator3D
