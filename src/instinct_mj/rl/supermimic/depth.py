"""Processed depth images of the task's camera from raw sensor measurements (real or rendered, see ``render.py``).

The camera pipeline is ``noise stages (meters, raw resolution) -> normalize -> crop_and_resize``. The policy's real
observation is the noised processed image; ``clean`` applies only the deterministic ``normalize`` + ``crop_and_resize`` to the
raw ``distance_to_image_plane`` (the sensor's own functions).

``noised`` is the imagination version of the sensor's noise pipeline (``depth_contour_noise``, ``depth_artifact_noise``,
``stereo_noise``, ``gaussian_blur_noise``, ``stereo_too_close_noise``, ``normalize``, ``crop_and_resize`` of
``_enable_depth_noise``), re-implemented as ONE stateless, host-sync-free function of the raw image that ``torch.compile`` fuses
(the sensor's functions use ``nonzero`` / boolean indexing / python ``random``, i.e. host syncs and dynamic shapes, and are
shared with the live sensor). It reads the stage parameters from the live sensor's cfgs and never calls or mutates the
sensor's noise objects. Equivalence (``tests/supermimic/check_depth_noise.py``): the deterministic stages (contour mask,
Gaussian blur, normalize, crop / resize) are the same function up to fp rounding; the random stages draw from the same
distributions:

- artifacts (``_add_depth_artifacts``): the original draws one Bernoulli(p) per pixel, then a rectangle size per hit, and
  renders every hit with ``grid_sample`` of a 25x25 box patch, which is SEPARABLE (a 1-D profile in y times one in x, both
  evaluated here in closed form). Here each image gets ``count ~ Binomial(H W, p)`` (inverse CDF of a fixed table, truncated
  where the tail is < 1e-7) artifacts at uniformly random pixels (with replacement) in a fixed number of slots; centers on
  pixels where the original's ``data > 0`` condition fails are dropped (thinning of uniform points = uniform over the valid
  ones). The per-image masks are summed and clamped as in the original;
- ``stereo_too_close``: the original shares ONE python-``random`` ``full_block_values`` choice per call (batch); here every
  image draws its own value (same marginal, independent across images) and the model runs on every image (images without a
  close pixel have empty masks, i.e. are left untouched, as in the original's pre-selection).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from instinct_mj.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthContourNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    StereoTooCloseNoiseCfg,
)

PATCH = 25  # side of the box patch of ``_add_depth_artifacts`` (1 inside [1, 23], 0 on the border ring)
STAGES = (DepthContourNoiseCfg, DepthArtifactNoiseCfg, RangeBasedGaussianNoiseCfg, GaussianBlurNoiseCfg, StereoTooCloseNoiseCfg,
          DepthNormalizationCfg, CropAndResizeCfg)  # fmt: skip
# contour kernels of ``depth_contour_noise``: (coefficient, (dy, dx) of the neighbor that is subtracted from the center)
CONTOUR = ((0.5, (-1, -1)), (0.1, (-1, 0)), (0.5, (-1, 1)), (1.2, (0, -1)), (1.2, (0, 1)), (0.5, (1, -1)), (0.1, (1, 0)), (0.5, (1, 1)))


def binomial_cdf(n: int, p: float, tail: float = 1e-7) -> torch.Tensor:
    """``cdf[j] = P(Binomial(n, p) <= j)`` for ``j < K``, K the smallest count with ``P(count > K) < tail``."""
    pmf = lambda j: math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log1p(-p))  # noqa: E731
    cdf, total = [], 0.0
    while True:
        total += pmf(len(cdf))
        cdf.append(total)
        if 1 - total < tail:
            return torch.tensor(cdf)


def box_profile(c: torch.Tensor, size: torch.Tensor, length: int) -> torch.Tensor:
    """1-D factor (..., length) of the ``grid_sample`` of the box patch stretched over ``[c - size/2, c + size/2]`` (both ends
    clipped to ``[0, length]``) along one image axis: the same grid, ``align_corners=False``, bilinear, zero padding."""
    lo, hi = (c - size / 2).clamp(0.0, float(length)), (c + size / 2).clamp(0.0, float(length))
    lin = torch.linspace(-1, 1, length, device=c.device)
    extent = hi - lo
    ok = extent > 0  # an empty rectangle covers nothing (the original's 0/0 grid samples nothing)
    g = (lin * length + length - hi.unsqueeze(-1) - lo.unsqueeze(-1)) / torch.where(ok, extent, 1.0).unsqueeze(-1)
    ix = ((g + 1) * PATCH - 1) / 2
    i0 = ix.floor()
    inside = lambda i: ((i >= 1) & (i <= PATCH - 2)).float()  # noqa: E731
    w1 = ix - i0
    return torch.where(ok.unsqueeze(-1), (1 - w1) * inside(i0) + w1 * inside(i0 + 1), 0.0)


class DepthPipeline:
    def __init__(self, env):
        self.cam = env.scene["camera"]
        pipeline = list(self.cam.noise_pipeline)
        assert tuple(type(c) for c in pipeline) == STAGES
        self.contour_cfg, self.artifact_cfg, self.stereo_cfg, self.blur_cfg, self.close_cfg, self.normalize_cfg, self.crop_cfg = pipeline
        self.device = self.contour_cfg.device  # set by the sensor to its own device
        self.H, self.W = self.cam.cfg.pattern.height, self.cam.cfg.pattern.width
        assert self.contour_cfg.maxpool_kernel_size == 1 and self.blur_cfg.kernel_size == 3
        assert self.normalize_cfg.normalize and self.crop_cfg.resize_shape is not None
        self.artifact_cdf = binomial_cdf(self.H * self.W, self.artifact_cfg.artifacts_prob).to(self.device)
        self.close_cdf = binomial_cdf(self.H * self.W, self.close_cfg.full_block_artifacts_prob).to(self.device)
        x = torch.arange(-1.0, 2.0) * 1.0  # torchvision's GaussianBlur 1-D kernel for kernel_size 3
        k1 = torch.exp(-0.5 * (x / self.blur_cfg.sigma) ** 2)
        self.blur_w = (k1 / k1.sum()).tolist()
        self.close_values = torch.tensor(self.close_cfg.full_block_values, device=self.device)

    def clean(self, meters: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) raw meters -> normalized + cropped / resized (B, 1, 18, 32), without the noise stages."""
        ids = torch.arange(meters.shape[0], device=meters.device)
        x = self.normalize_cfg.func(meters, self.normalize_cfg, ids)
        return self.crop_cfg.func(x, self.crop_cfg, ids).permute(0, 3, 1, 2)

    def contour_mask(self, d: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> bool mask of ``depth_contour_noise``: max over the 8 kernels of |c (center - neighbor)| > threshold
        (zero padding: a neighbor outside the image is 0)."""
        p = F.pad(d, (1, 1, 1, 1))
        H, W = d.shape[-2:]
        m = torch.zeros_like(d)
        for coef, (dy, dx) in CONTOUR:
            m = torch.maximum(m, (coef * (d - p[:, 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W])).abs())
        return m > self.contour_cfg.contour_threshold

    def blur(self, d: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> ``gaussian_blur_noise`` (3x3, reflect padding, separable weights of torchvision's ``GaussianBlur``)."""
        w0, w1, w2 = self.blur_w
        p = F.pad(d.unsqueeze(1), (1, 1, 1, 1), mode="reflect")[:, 0]
        r = w0 * p[:, :, :-2] + w1 * p[:, :, 1:-1] + w2 * p[:, :, 2:]
        return w0 * r[:, :-2] + w1 * r[:, 1:-1] + w2 * r[:, 2:]

    def artifact_mask(self, cdf: torch.Tensor, prob_valid: torch.Tensor | None, hm_std, wm_std, n: int) -> torch.Tensor:
        """(n, H, W) union (sum clamped to [0, 1]) of the random rectangle artifacts of ``_add_depth_artifacts``;
        ``prob_valid`` (n, H, W) bool = the pixels allowed as centers (``data > 0``), None = all."""
        H, W, dev = self.H, self.W, self.device
        K = len(cdf)
        count = (torch.rand(n, 1, device=dev) > cdf).sum(1, keepdim=True)  # (n, 1) in 0..K
        active = torch.arange(K, device=dev) < count  # (n, K)
        cy, cx = (torch.rand(2, n, K, device=dev) * torch.tensor([[[H]], [[W]]], device=dev)).floor()
        if prob_valid is not None:
            active = active & prob_valid.flatten(1).gather(1, (cy * W + cx).long())
        sh = (hm_std[0] + torch.randn(n, K, device=dev) * hm_std[1]).clamp(0.0, float(H))
        sw = (wm_std[0] + torch.randn(n, K, device=dev) * wm_std[1]).clamp(0.0, float(W))
        fy, fx = (box_profile(c, z, L) * active.unsqueeze(-1) for c, z, L in ((cy, sh, H), (cx, sw, W)))
        return torch.einsum("nkh,nkw->nhw", fy, fx).clamp(0.0, 1.0)

    def noised(self, meters: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 1) raw meters -> the policy's observation of the image (B, 1, 18, 32): the sensor's whole noise pipeline
        (new random samples on every call; differentiable w.r.t. ``meters``; stateless, see the module docstring)."""
        d = meters[..., 0]
        n = d.shape[0]
        a, s, c = self.artifact_cfg, self.stereo_cfg, self.close_cfg
        with torch.no_grad():
            mask = self.contour_mask(d)
        d = torch.where(mask, torch.zeros_like(d), d)
        # depth_artifact_noise
        m = self.artifact_mask(self.artifact_cdf, d.detach() > 0, a.artifacts_height_mean_std, a.artifacts_width_mean_std, n)
        d = d * (1 - m) + m * a.noise_value
        # stereo_noise (range based Gaussian)
        apply = (d.detach() >= s.min_value) & (d.detach() <= s.max_value)
        d = d + torch.randn_like(d) * s.noise_std * apply
        d = self.blur(d)  # gaussian_blur_noise
        # stereo_too_close_noise
        close = d.detach() < c.close_threshold
        vertical = close.sum(1, keepdim=True) > (self.H * 0.6)
        full, half = vertical & close, (~vertical) & close
        value = self.close_values[torch.randint(len(self.close_values), (n, 1, 1), device=self.device)]
        art = self.artifact_mask(self.close_cdf, None, c.full_block_height_mean_std, c.full_block_width_mean_std, n)
        spark = torch.rand_like(d) < c.half_block_spark_prob
        d = torch.where(full, art * value, d)
        d = torch.where(half, spark.float() * c.half_block_value, d)
        # normalize + crop_and_resize
        nz, cr = self.normalize_cfg, self.crop_cfg
        d = d.clip(nz.depth_range[0], nz.depth_range[1])
        d = (d - nz.depth_range[0]) / (nz.depth_range[1] - nz.depth_range[0])
        d = d * (nz.output_range[1] - nz.output_range[0]) + nz.output_range[0]
        up, down, left, right = cr.crop_region
        d = d[:, up : self.H - down, left : self.W - right]
        return F.interpolate(d.unsqueeze(1), size=cr.resize_shape, mode="bilinear", align_corners=False)

    def clean_from_sensor(self) -> torch.Tensor:
        """Clean processed depth image of every env (B, 1, 18, 32) from the sensor's latest raw measurement."""
        return self.clean(self.cam.data.output["distance_to_image_plane"].clone())
