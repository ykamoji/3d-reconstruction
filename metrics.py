import torch
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()

    fallback = None
    if sigma1.device.type == "mps":
        fallback = sigma1.device.type
        sigma1 = sigma1.cpu()
        sigma2 = sigma2.cpu()

    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    if fallback:
        c = c.to(fallback)

    return a + b - 2 * c


class FrechetInceptionDistanceModified(FrechetInceptionDistance):

    def __int__(self, args, kwargs):
        super(FrechetInceptionDistanceModified, self).__init__(**args, **kwargs)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        if features.device.type != "mps":
            features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)


def compute_flatness_score(depths, num_bins=64, min_depth=64, max_depth=64):

    depth_histograms = convert_depth_maps_to_histograms(depths, bins=num_bins, min=min_depth, max=max_depth)

    entropy = compute_histogram_entropy(depth_histograms)
    flatness_score = entropy.exp().mean().item()

    return float(flatness_score)


@torch.no_grad()
def convert_depth_maps_to_histograms(depth_maps: torch.Tensor, *args, **kwargs):
    """
    Unfortunately, torch cannot compute histograms batch-wise...
    """
    histograms = torch.stack([torch.histc(d, *args, **kwargs) for d in depth_maps], dim=0) # [num_depth_maps, num_bins]
    # Validating the histograms
    counts = histograms.sum(dim=1) # [num_depth_maps]
    assert counts.min() == counts.max() == depth_maps[0].numel(), f"Histograms countain OOB values: {counts.min(), counts.max(), depth_maps.shape}"

    return histograms


def compute_histogram_entropy(histograms: torch.Tensor) -> torch.Tensor:
    assert histograms.ndim == 2, f"Wrong shape: {histograms.shape}"
    probs = histograms / histograms.sum(dim=1, keepdim=True) # [batch_size, num_bins]
    return -1.0 * (torch.log(probs + 1e-12) * probs).sum(dim=1)