"""Read-only climb dynamics and contact audit, using ``check_dynamics``.

Example::

  uv run --no-sync python -m instinct_mj.scripts.acp.analyze_climb \
    /home/duanxin/Datasets/acp/v3_train_boxclimb_20261112_v1 \
    --workers 4 --out /tmp/climb_audit

All saved frames are audited. Root wrench feasibility is a necessary condition,
not a policy tracking success prediction. The 20 mm contact allowance is relaxed;
the existing 2 mm and friction sensitivity results are retained alongside it.
Label-restricted and geometry-only LPs use the same contacts: their difference
isolates the effect of excluding contacts that have no positive sidecar label.
Their 0.05 residual difference is descriptive, not a calibrated failure rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from instinct_mj.scripts.acp import check_dynamics as dyn

CLIMB_DIRS = ("box_", "climb_", "low-climb", "high-climb", "high-step")


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_summary(z, name: str) -> tuple[dict, list[dict]]:
    """Summarize saved LP results; percentages use valid frames only."""
    valid, scores = z["valid"].astype(bool), z["scores"]
    dt = float(z["dt"])
    rising = valid & (np.gradient(z["qpos"][:, 2], dt) > .15)
    gap = scores[:, 6] - scores[:, 0]
    label_limited = valid & np.isfinite(gap) & (gap > .05)

    def mean(mask, selection=valid):
        return float(np.mean(mask[selection])) if selection.any() else None

    def p90(x, selection=valid):
        return float(np.quantile(x[selection], .9)) if selection.any() else None

    runs = dyn.intervals(label_limited, min_length=5)
    summary = dict(
        clip=name, frames=int(valid.sum()), rising_frames=int(rising.sum()),
        root_residual_p90=p90(scores[:, 0]),
        root_residual_strict_2mm_p90=p90(scores[:, 4]),
        root_residual_mu05_p90=p90(scores[:, 2]),
        label_root_residual_p90=p90(scores[:, 6]),
        rising_root_residual_p90=p90(scores[:, 0], rising),
        rising_label_root_residual_p90=p90(scores[:, 6], rising),
        label_limited_frame_fraction=mean(label_limited),
        label_limited_intervals=runs,
        label_limited_intervals_seconds=[[a * dt, (b + 1) * dt] for a, b in runs],
        rising_without_any_label_fraction=mean(~z["labels"].any(1), rising),
        rising_without_hand_label_fraction=mean(~z["labels"][:, 7:9].any(1), rising),
        required_peak_torque_ratio_p90=p90(z["torque"][:, 0]),
        required_peak_torque_ratio_max=float(z["torque"][valid, 0].max()),
    )
    contacts = []
    for j, body in enumerate(dyn.CONTACT_LINKS):
        flag = z["labels"][:, j].astype(bool)
        labelled = valid & flag
        near = valid & (z["distance"][:, j] <= .002)
        onset = flag & ~np.r_[True, flag[:-1]]
        contacts.append(dict(
            clip=name, body=body, labelled_frames=int(labelled.sum()),
            label_onsets=int((valid & onset).sum()),
            label_gap_p90_mm=p90(1000 * z["distance"][:, j], labelled),
            label_gap_over_20mm_fraction=mean(z["distance"][:, j] > .02, labelled),
            near_2mm_unlabelled_frames=int((near & ~flag).sum()),
            near_unlabelled_intervals=dyn.intervals(near & ~flag, min_length=5),
        ))
    return summary, contacts


def run_job(entry, out):
    return dyn.audit_clip(entry, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-regex", help="Optional further selection by relative path or group")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--calibration", type=Path, help="Optional verified-rollout calibration JSON")
    parser.add_argument("--resume", action="store_true", help="Reuse completed frames only with an identical input/code manifest")
    args = parser.parse_args()
    root, out = args.dataset.expanduser().resolve(), args.out.expanduser().resolve()
    selected = []
    for entry in dyn.discover(root):
        rel = str(Path(entry["path"]).relative_to(root))
        if not rel.split("/")[0].startswith(CLIMB_DIRS):
            continue
        if args.clip_regex and not re.search(args.clip_regex, rel + " " + entry["group"]):
            continue
        selected.append(entry)
    if not selected:
        parser.error("No box/climb motions matched the selection")
    manifest = dict(dataset=str(root), metadata_sha=file_sha(root / "metadata.yaml"),
                    code_sha=file_sha(__file__), helper_sha=file_sha(dyn.__file__),
                    asset_code_sha=file_sha(Path(dyn.get_acp_g1_spec.__code__.co_filename)),
                    inputs={e["name"]: {str(p): file_sha(p) for p in
                        [e["path"], str(Path(e["path"]).with_suffix(".contacts.npz")), e["terrain"]]}
                        for e in selected})
    if (out / "manifest.json").exists():
        if not args.resume or json.loads((out / "manifest.json").read_text()) != manifest:
            parser.error("Output exists; --resume requires identical selected input files and audit code")
    elif out.exists() and any(out.iterdir()):
        parser.error("Refusing an existing nonempty output directory without an audit manifest")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "selection.json").write_text(json.dumps(selected, indent=2))
    pending = [e for e in selected if not ((out / "frames" / (e["name"] + ".npz")).exists()
                                         and (out / "frames" / (e["name"] + ".json")).exists())]
    print(f"Selected {len(selected)} clips, {len(pending)} pending; CPU workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(run_job, e, out): e for e in pending}
        for i, future in enumerate(as_completed(jobs), 1):
            info = future.result()
            print(f"{i}/{len(pending)} {info['name']}: {info['frames']} frames", flush=True)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    dyn.summarize(out, calibration)
    climbs, contacts = [], []
    for entry in selected:
        with np.load(out / "frames" / (entry["name"] + ".npz")) as z:
            row, rows = contact_summary(z, entry["name"])
        row["motion_file"] = str(Path(entry["path"]).relative_to(root))
        climbs.append(row)
        contacts.extend(rows)
    dyn.write_csv(out / "climb.csv", climbs)
    dyn.write_csv(out / "contact_diagnostics.csv", contacts)
    result = dict(dataset=str(root), clips=len(climbs),
                  calibrated=bool(calibration and calibration.get("calibrated")),
                  caveats=["Necessary wrench/torque feasibility only; not closed-loop tracking validation",
                           "20 mm contacts are relaxed; inspect the 2 mm sensitivity alongside them",
                           "Unlabelled near contacts need phase/velocity review; proximity alone is not support",
                           "No artificial bound on normal force; hand grip/tensile support is not modelled"],
                  clip_results=climbs, contact_results=contacts)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote {out}/summary.json, climb.csv, contact_diagnostics.csv", flush=True)


if __name__ == "__main__":
    main()
