"""Depth history/skip-frame probe for perceptive task alignment."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import instinct_mj.tasks  # noqa: F401
from instinct_mj.envs import InstinctRlEnv
from instinct_mj.tasks.registry import list_tasks, load_env_cfg
from instinct_mj.utils.motion_validation import validate_tracking_motion_file


@dataclass(frozen=True)
class ProbeConfig:
    motion_file: str | None = None
    registry_name: str | None = None
    steps: int = 16
    num_envs: int | None = 64
    device: str | None = None
    baseline_json: str | None = None
    output_json: str | None = None
    compare_abs_tol: float = 0.03
    compare_rel_tol: float = 0.15


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_tracking_motion(task_id: str, cfg: ProbeConfig, env_cfg) -> None:
    is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    if not is_tracking_task:
        return

    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if cfg.motion_file is not None:
        motion_path = Path(cfg.motion_file).expanduser().resolve()
        validate_tracking_motion_file(motion_path)
        motion_cmd.motion_file = str(motion_path)
        return

    if cfg.registry_name:
        registry_name = cfg.registry_name
        if ":" not in registry_name:
            registry_name = registry_name + ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_path = (Path(artifact.download()) / "motion.npz").resolve()
        validate_tracking_motion_file(motion_path)
        motion_cmd.motion_file = str(motion_path)
        return

    configured_motion = str(getattr(motion_cmd, "motion_file", "")).strip()
    if configured_motion:
        configured_path = Path(configured_motion).expanduser().resolve()
        validate_tracking_motion_file(configured_path)
        motion_cmd.motion_file = str(configured_path)
        print(f"[INFO] Using motion file from env config: {configured_path}")
        return

    raise ValueError("Tracking probe requires a motion file.\n  --motion-file /path/to/motion.npz")


def _tensor_stats(data: torch.Tensor) -> dict[str, float | list[int]]:
    x = data.detach().float()
    percentiles = torch.quantile(x.reshape(-1), torch.tensor([0.01, 0.5, 0.99], device=x.device))
    stats: dict[str, float | list[int]] = {
        "shape": list(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p01": float(percentiles[0].item()),
        "p50": float(percentiles[1].item()),
        "p99": float(percentiles[2].item()),
    }
    return stats


def _aggregate_metrics(
    records: list[dict[str, float | list[int] | int]],
) -> dict[str, float]:
    metric_names = [
        "min",
        "max",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "history_delta_mean",
        "history_delta_max",
    ]
    aggregated: dict[str, float] = {}
    for metric in metric_names:
        values: list[float] = []
        for record in records:
            value = record.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            aggregated[metric] = float(sum(values) / len(values))
    return aggregated


def _compare_with_baseline(
    *,
    current_summary: dict,
    baseline_summary: dict,
    abs_tol: float,
    rel_tol: float,
) -> dict:
    current_metrics = _aggregate_metrics(current_summary.get("records", []))
    baseline_metrics = _aggregate_metrics(baseline_summary.get("records", []))
    all_metrics = sorted(set(current_metrics.keys()) & set(baseline_metrics.keys()))

    metric_results: dict[str, dict[str, float | bool]] = {}
    overall_pass = True
    for metric in all_metrics:
        current_value = current_metrics[metric]
        baseline_value = baseline_metrics[metric]
        abs_diff = abs(current_value - baseline_value)
        denom = max(abs(baseline_value), 1.0e-6)
        rel_diff = abs_diff / denom
        threshold = max(abs_tol, rel_tol * denom)
        passed = abs_diff <= threshold
        overall_pass = overall_pass and passed
        metric_results[metric] = {
            "current": current_value,
            "baseline": baseline_value,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "threshold": threshold,
            "pass": passed,
        }

    current_shape = None
    baseline_shape = None
    if current_summary.get("records"):
        current_shape = current_summary["records"][0].get("shape")
    if baseline_summary.get("records"):
        baseline_shape = baseline_summary["records"][0].get("shape")
    shape_match = current_shape == baseline_shape
    overall_pass = overall_pass and shape_match

    return {
        "pass": overall_pass,
        "shape_match": shape_match,
        "current_shape": current_shape,
        "baseline_shape": baseline_shape,
        "metrics": metric_results,
    }


def run_probe(task_id: str, cfg: ProbeConfig) -> None:
    configure_torch_backends()
    device = _resolve_device(cfg.device)
    env_cfg = load_env_cfg(task_id)
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    _resolve_tracking_motion(task_id, cfg, env_cfg)

    env = InstinctRlEnv(cfg=env_cfg, device=device, render_mode=None)
    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim

    step_stats: list[dict[str, float | list[int] | int]] = []
    for step in range(cfg.steps):
        policy_obs = obs["policy"]
        depth = policy_obs.get("depth_image") if isinstance(policy_obs, dict) else None

        record: dict[str, float | list[int] | int] = {"step": step}
        if isinstance(depth, torch.Tensor):
            record.update(_tensor_stats(depth))
            if depth.ndim >= 4 and depth.shape[1] > 1:
                temporal_delta = (depth[:, 1:] - depth[:, :-1]).abs()
                record["history_delta_mean"] = float(temporal_delta.mean().item())
                record["history_delta_max"] = float(temporal_delta.max().item())
        else:
            record["depth_image_missing"] = 1
        step_stats.append(record)

        actions = torch.zeros((env.num_envs, action_dim), device=device, dtype=torch.float32)
        obs, _, _, _, _ = env.step(actions)

    env.close()
    summary = {
        "task": task_id,
        "device": device,
        "num_envs": env_cfg.scene.num_envs,
        "steps": cfg.steps,
        "policy_terms": list(env_cfg.observations["policy"].terms.keys()),
        "critic_terms": list(env_cfg.observations["critic"].terms.keys()),
        "records": step_stats,
    }

    if cfg.baseline_json is not None:
        baseline_path = Path(cfg.baseline_json).expanduser().resolve()
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline depth json not found: {baseline_path}")
        baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
        summary["comparison"] = _compare_with_baseline(
            current_summary=summary,
            baseline_summary=baseline_summary,
            abs_tol=cfg.compare_abs_tol,
            rel_tol=cfg.compare_rel_tol,
        )

    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if cfg.output_json is not None:
        output_path = Path(cfg.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_text, encoding="utf-8")
        print(f"[INFO] Wrote probe json: {output_path}")

    print(summary_text)


def main() -> None:
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg = tyro.cli(
        ProbeConfig,
        args=remaining_args,
        default=ProbeConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_probe(task_id=chosen_task, cfg=cfg)


if __name__ == "__main__":
    main()
