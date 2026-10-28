"""Equivalence of ``DepthPipeline.noised`` (fused, stateless) with the sensor's own noise pipeline.

    .venv/bin/python tests/supermimic/check_depth_noise.py

Deterministic stages are compared on the live sensor's raw images against the sensor's functions (contour mask, blur,
normalize + crop / resize: fp rounding); the separable closed form of the box artifacts against the original ``grid_sample``
construction on the same random rectangles; and the whole random pipeline in distribution (per-pixel mean / std / quantiles
over many samples of the same raw images, with and without very close pixels)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from instinct_mj.scripts.supermimic.common import TASK_ID, build_env_cfg, make_vec_env

from instinct_mj.rl.supermimic.depth import PATCH, DepthPipeline, box_profile

dev = "cuda:0"
cfg = build_env_cfg(TASK_ID, 64, "~/Datasets/20251116_50cm_kneeClimbStep1")
cfg.seed = 0
env = make_vec_env(TASK_ID, cfg, dev).unwrapped
pipe = DepthPipeline(env)
cam = env.scene["camera"]
env.reset()
n_act = env.action_manager.total_action_dim
for _ in range(40):
    env.step(0.3 * torch.randn(64, n_act, device=dev))
raw = cam.data.output["distance_to_image_plane"].clone()  # (64, H, W, 1)
ids = torch.arange(64, device=dev)
H, W = pipe.H, pipe.W
print("raw", tuple(raw.shape), "min %.3f max %.3g" % (raw.min(), raw.max()))

# ---- deterministic stages ----
d0 = raw[..., 0]
ref = pipe.contour_cfg.func(raw.clone(), pipe.contour_cfg, ids)[..., 0]
mine = torch.where(pipe.contour_mask(d0), torch.zeros_like(d0), d0)
print("contour: max |diff| %.3g, masked pixels %.4f" % ((ref - mine).abs().max(), (ref != d0).float().mean()))
x = torch.rand(64, H, W, 1, device=dev) * 2
ref = pipe.blur_cfg.func(x.clone(), pipe.blur_cfg, ids)[..., 0]
print("blur: max |diff| %.3g" % (ref - pipe.blur(x[..., 0])).abs().max())

# ---- separable box artifacts vs the original grid_sample construction ----
n = 4096
cy, cx = torch.randint(0, H, (n,), device=dev).float(), torch.randint(0, W, (n,), device=dev).float()
sh, sw = (2 + 0.5 * torch.randn(n, device=dev)).clamp(0, H), (3 + 3 * torch.randn(n, device=dev)).clamp(0.5, W)
top, bottom, left, right = (cy - sh / 2).clamp(0, H), (cy + sh / 2).clamp(0, H), (cx - sw / 2).clamp(0, W), (cx + sw / 2).clamp(0, W)
patch = torch.zeros(n, 1, PATCH, PATCH, device=dev)
patch[:, :, 1:24, 1:24] = 1.0
grid = torch.zeros(n, H, W, 2, device=dev)
grid[..., 0] = torch.linspace(-1, 1, W, device=dev).view(1, 1, W)
grid[..., 1] = torch.linspace(-1, 1, H, device=dev).view(1, H, 1)
grid[..., 0] = (grid[..., 0] * W + W - right[:, None, None] - left[:, None, None]) / (right - left)[:, None, None]
grid[..., 1] = (grid[..., 1] * H + H - bottom[:, None, None] - top[:, None, None]) / (bottom - top)[:, None, None]
orig = F.grid_sample(patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)
sep = box_profile(cy, sh, H).unsqueeze(-1) * box_profile(cx, sw, W).unsqueeze(-2)
print("box artifact: max |diff| %.3g (values up to %.2f)" % ((orig - sep).abs().max(), orig.max()))

# a rectangle of size 0 covers nothing (the original's 0/0 grid), no NaN anywhere in the pipeline
z = box_profile(torch.tensor([3.0, 0.0, 26.0], device=dev), torch.zeros(3, device=dev), H)
print("empty rectangle profile: max %.3g, nan %s" % (z.max(), bool(torch.isnan(z).any())))
nan = sum(bool(torch.isnan(pipe.noised(raw.repeat(16, 1, 1, 1))).any()) for _ in range(100))
print("noised() calls with NaN out of 100 (1024 images each): %d" % nan)

# ---- whole pipeline in distribution ----
def sample(fn, images, reps):
    out = torch.stack([fn(images) for _ in range(reps)])  # (reps, B, 1, 18, 32)
    return out


def compare(name, images, reps=200):
    a = sample(lambda r: cam.apply_noise_pipeline(r.clone(), ids).permute(0, 3, 1, 2), images, reps)
    b = sample(pipe.noised, images, reps)
    se = (a.var(0) / reps + b.var(0) / reps).sqrt().clamp(min=1e-4)
    z = (a.mean(0) - b.mean(0)) / se
    qs = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=dev)
    qa, qb = torch.quantile(a.flatten(), qs), torch.quantile(b.flatten(), qs)
    print(f"{name}: mean {a.mean():.5f} vs {b.mean():.5f}; std {a.std():.5f} vs {b.std():.5f}; frac==0 {(a == 0).float().mean():.5f} vs {(b == 0).float().mean():.5f}; "
          f"frac==1 {(a == 1).float().mean():.5f} vs {(b == 1).float().mean():.5f}")
    print(f"   per-pixel mean z-score: rms {z.pow(2).mean().sqrt():.3f} (1 = same distribution), max {z.abs().max():.2f}; quantiles {qa.tolist()} vs {qb.tolist()}")
    print(f"   per-pixel std ratio mine/sensor (pixels with std > 1e-3): mean {(b.std(0) / a.std(0).clamp(min=1e-6))[a.std(0) > 1e-3].mean():.3f}")


compare("sensor images", raw)
close = raw.clone()
close[:32, :, 10:14] = 0.05  # some columns very close
close[32:, : H // 2 + 5, :] = 0.08  # most of the image very close (vertical blocks)
compare("with close pixels", close)
pipe.close_cfg.full_block_values = [1.0]  # the original shares one value per call across the batch: pin it to compare tightly
pipe.close_values = torch.tensor([1.0], device=dev)
compare("with close pixels, single full-block value", close, reps=400)
