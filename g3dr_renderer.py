import torch
from random import random
import torch.nn.functional as F
from eg3d.renderer import ImportanceRenderer, sample_from_planes, sample_from_3dgrid, generate_planes, math_utils, \
    project_onto_planes

GRAD_CLIP_MIN = 0.05
STYLE = 'lod_no'


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, sigmas, scaling):
        ctx.save_for_backward(scaling)
        return colors, sigmas, scaling

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
        (scaling,) = ctx.saved_tensors
        scaling = scaling.clamp(GRAD_CLIP_MIN, 1)
        return grad_output_colors * scaling, grad_output_sigmas * scaling, grad_output_ray_dist


class extend_MipRayMarcher(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colors, densities, depths, rendering_options, grad_scaling=None):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1)  # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        if type(grad_scaling) != type(None):
            alpha, colors_mid, grad_scaling = GradientScaler.apply(alpha, colors_mid, grad_scaling)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / (weight_total + + 0.001)

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1  # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights, weight_total


class ImportanceRenderer_extended(ImportanceRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ray_marcher = extend_MipRayMarcher()
        self.STYLE = STYLE
        self.gauss_pdf = lambda x, mean, std: 1.25 * torch.exp(- ((x - mean) ** 2) / std)

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, importance_depth=None):
        with torch.no_grad():
            self.plane_axes = self.plane_axes.to(planes.device)

            if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
                ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions,
                                                                   box_side_length=rendering_options['box_warp'])
                is_ray_valid = ray_end > ray_start
                if torch.any(is_ray_valid).item():
                    ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                    ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
                depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end,
                                                       rendering_options['depth_resolution'],
                                                       rendering_options['disparity_space_sampling'])
            else:
                # Create stratified depth samples
                depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'],
                                                       rendering_options['ray_end'],
                                                       rendering_options['depth_resolution'],
                                                       rendering_options['disparity_space_sampling'])

            batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(
                batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size,
                                                                                                         -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            depths_fine = None
            with torch.no_grad():
                if type(importance_depth) != type(None):
                    bs = importance_depth.shape[0]
                    importance_depth_reshape = importance_depth.permute(0, 2, 3, 1).reshape(bs, -1, 1)
                    importance_depth_reshape = importance_depth_reshape[:, :, None, :]
                    if random() > 0.6:
                        var = 0.05
                        sample_uniform = torch.linspace(-var, var, N_importance).to(planes.device)
                        depths_fine = importance_depth_reshape + sample_uniform[None, None, :, None]
                if depths_fine is None:
                    _, _, weights, weight_total = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse,
                                                                   rendering_options)
                    depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

                sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(
                    batch_size, -1, 3)
                sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine
                                      * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse,
                                                                       densities_coarse,
                                                                       depths_fine, colors_fine, densities_fine)

            # Aggregate
            output_depth = all_depths
            depths_mid = (output_depth[:, :, :-1] + output_depth[:, :, 1:]) / 2

            if type(importance_depth) != type(None):
                grad_scaling = self.gauss_pdf(depths_mid, importance_depth_reshape, 0.03)
            else:
                grad_scaling = None
            rgb_final, depth_final, weights, weight_total = self.ray_marcher(all_colors, all_densities, all_depths,
                                                                             rendering_options, grad_scaling)
        else:
            output_depth = depths_coarse
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse,
                                                               rendering_options)
        return rgb_final, depth_final, weight_total, depths_mid

    def sample_from_planes_hie(self, plane_axes, plane_features1, plane_features2, plane_features3, coordinates,
                               mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        _, M, _ = coordinates.shape
        N, n_planes, C, H, W = plane_features1.shape
        plane_features1 = plane_features1.view(N * n_planes, C, H, W)
        N, n_planes, C, H, W = plane_features2.shape
        plane_features2 = plane_features2.view(N * n_planes, C, H, W)
        N, n_planes, C, H, W = plane_features3.shape
        plane_features3 = plane_features3.view(N * n_planes, C, H, W)

        coordinates = (2 / box_warp) * coordinates  # TODO: add specific box bounds
        with torch.no_grad():
            projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).float()
        output_features = torch.nn.functional.grid_sample(plane_features1, projected_coordinates, mode=mode,
                                                          padding_mode=padding_mode,
                                                          align_corners=False)  # .permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features += torch.nn.functional.grid_sample(plane_features2, projected_coordinates, mode=mode,
                                                           padding_mode=padding_mode,
                                                           align_corners=False)  # .permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        output_features += torch.nn.functional.grid_sample(plane_features3, projected_coordinates, mode=mode,
                                                           padding_mode=padding_mode, align_corners=False)
        output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        if self.STYLE == 'lod_no':
            planes_reshape = planes.view(planes.shape[0], -1, planes.shape[-2], planes.shape[-1])

            MPS_FIX = False
            if planes_reshape.device.type == 'mps':
                planes_reshape = planes_reshape.cpu()
                MPS_FIX = True

            planes_64 = F.interpolate(planes_reshape, scale_factor=0.5, mode="bilinear", antialias=True)
            if MPS_FIX:
                planes_64 = planes_64.to('mps')

            planes_64 = planes_64.view(planes.shape[0], 3, -1, planes.shape[-2] // 2, planes.shape[-1] // 2)

            if MPS_FIX:
                planes_reshape = planes_reshape.cpu()

            planes_32 = F.interpolate(planes_reshape, scale_factor=0.25, mode="bilinear", antialias=True)

            if MPS_FIX:
                planes_32 = planes_32.to('mps')

            planes_32 = planes_32.view(planes.shape[0], 3, -1, planes.shape[-2] // 4, planes.shape[-1] // 4)

            planes_128 = planes

            sampled_features = self.sample_from_planes_hie(self.plane_axes, planes_128, planes_64, planes_32,
                                                           sample_coordinates, padding_mode='zeros',
                                                           box_warp=options['box_warp'])
            sampled_features = sampled_features.mean(1, keepdims=True)
        else:
            sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros',
                                                  box_warp=options['box_warp'])
            sampled_features = sampled_features.mean(1, keepdims=True)

        coordinates = (2 / options['box_warp']) * sample_coordinates
        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out
