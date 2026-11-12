"""Refresh contact geometry using the original contact bodies and timing.

Writes a NEW snapshot. Input motions, terrain assets and original sidecars are
never overwritten. Contact flags are copied by link name; no support inference,
duration filter or distance threshold changes the reference sequence. Current
body surface points nearest each original contact anchor are projected onto the
terrain, then onset targets are rebuilt on the ORIGINAL segments. Unsupported
contacts and changes of terrain face are exported for motion repair/review.

Example::

  uv run --no-sync python -m instinct_mj.scripts.acp.refresh_contacts \
    --dataset /path/to/current --reference-dataset /path/to/original \
    --out /path/to/new_snapshot --simplified-collision --workers 3
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from instinct_mj.assets.acp_g1 import ACP_G1_XML, get_acp_g1_spec
from instinct_mj.rl.acp.planner import CONTACT_BODIES, CONTACT_LINKS
from instinct_mj.scripts.acp import label_contacts as lab
from instinct_mj.scripts.acp.check_dynamics import write_csv

_LABELER = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_arrays(reference: dict, frames: int, fps: float):
    names = [str(n) for n in reference["link_names"]]
    if len(names) != len(set(names)) or set(names) != set(CONTACT_LINKS):
        raise ValueError("Reference must contain each CONTACT_LINKS entry exactly once")
    order = [names.index(n) for n in CONTACT_LINKS]
    flags = np.asarray(reference["contact_flag"])
    pos = np.asarray(reference["contact_pos_w"])
    if flags.shape != (frames, len(names)) or pos.shape != (frames, len(names), 3):
        raise ValueError("Reference contacts do not match the current motion length")
    if not np.isin(flags, [0, 1]).all() or not np.isfinite(pos).all():
        raise ValueError("Invalid reference flag or contact position")
    if not np.isclose(float(reference["framerate"]), fps, rtol=0, atol=1e-6):
        raise ValueError("Reference frame rate differs; explicit time alignment is required")
    return flags[:, order].astype(bool), pos[:, order].copy()


def assemble(features: dict, flag: np.ndarray, params: dict) -> dict:
    """Keep the original sequence, including short/skidding/temporarily distant contacts."""
    if flag.shape != features["min_dist"].shape:
        raise ValueError("Reference flag/features shape mismatch")
    normal = np.asarray(features["normal"], dtype=np.float64).copy()
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm[..., 0][flag] < 1e-10):
        raise ValueError("Zero normal at an original contact")
    # The BVH distance/closest-point query is float32. Its separation vector
    # and reported length can differ by ~4e-5 relatively around narrow rungs.
    normal /= np.maximum(norm, 1e-12)
    onset = np.zeros_like(features["pos"])
    for b in range(flag.shape[1]):
        for start, end in lab._runs(flag[:, b]):
            onset[start:end, b] = features["pos"][start, b]
    return dict(
        framerate=np.asarray(features["framerate"]), body_names=np.asarray(CONTACT_BODIES), link_names=np.asarray(CONTACT_LINKS),
        contact_flag=flag.copy(), contact_pos_w=np.where(flag[..., None], features["pos"], 0).astype(np.float32),
        onset_pos_w=onset.astype(np.float32),
        contact_normal_w=np.where(flag[..., None], normal, 0).astype(np.float32),
        min_dist=features["min_dist"].astype(np.float32), contact_speed=features["speed"].astype(np.float32),
        com_w=features["com"].astype(np.float32), support_added=np.zeros_like(flag),
        support_status=np.full(len(flag), lab.SUPPORT_NOT_CHECKED, dtype=np.int8),
        params=np.asarray(json.dumps(params, sort_keys=True)),
    )


def process(job: dict) -> dict:
    global _LABELER
    p, side, terrain, out = (Path(job[k]) for k in ("motion", "reference", "terrain", "out"))
    if _LABELER is None:
        _LABELER = lab.ContactLabeler(job["simplified_collision"])
    with np.load(p) as z:
        motion = {k: z[k] for k in z.files}
    with np.load(side) as z:
        reference = {k: z[k] for k in z.files}
    flag, anchor = reference_arrays(reference, len(motion["joint_pos"]), float(motion["framerate"]))
    old_params = json.loads(str(reference["params"])) if "params" in reference else {}
    params = lab.default_params()
    params.update({k: v for k, v in old_params.items() if k in params})
    params.update(simplified_collision=job["simplified_collision"], support=False,
                  source="original_sequence_current_geometry", reference_params=old_params,
                  position_source="current_body_surface_nearest_original_anchor_then_terrain_projection",
                  sequence_policy="original_flags_and_segment_boundaries_unchanged",
                  reference_contact_sha256=job["reference_sha256"], motion_sha256=job["motion_sha256"])
    _, distance = lab._get_terrain(str(terrain))
    features = _LABELER.extract(motion, distance, params, reference_positions=anchor, reference_flags=flag)
    labels = assemble(features, flag, params)
    if not np.array_equal(labels["contact_flag"], flag):
        raise RuntimeError("Reference contact sequence changed")
    for k, v in labels.items():
        if v.dtype.kind in "fc" and not np.isfinite(v).all():
            raise RuntimeError(f"Non-finite refreshed {k}: {p}")
    active_norm = np.linalg.norm(labels["contact_normal_w"][flag], axis=1)
    if active_norm.size and not np.allclose(active_norm, 1, atol=1e-5):
        raise RuntimeError(f"Invalid contact normal: {p}")
    projection_error, _ = distance.query(labels["contact_pos_w"][flag])
    if projection_error.size and projection_error.max() > .001:
        raise RuntimeError(f"Refreshed contact point is not on the terrain: {p}")
    delta = np.linalg.norm(labels["contact_pos_w"] - anchor, axis=-1)
    old_surface, _ = distance.query(anchor[flag])
    patch = features["contact_patch_dist"]
    npos = int(flag.sum())
    quantile = lambda x, q: float(np.quantile(x, q)) if x.size else 0.0
    # These thresholds only classify review items; they never change labels.
    review = flag & ((patch > .05) | (delta > .05) | features["surface_changed"])
    events = []
    for b, body in enumerate(CONTACT_LINKS):
        for start, end in lab._runs(review[:, b]):
            events.append(dict(body=body, first_frame=start, end_frame_exclusive=end,
                               max_patch_gap_mm=float(patch[start:end, b].max()*1000),
                               max_target_shift_mm=float(delta[start:end, b].max()*1000),
                               surface_changed=bool(features["surface_changed"][start:end, b].any())))
    report = dict(motion_file=job["rel"], frames=len(flag), positive_pairs=npos, changed_flag_pairs=0,
                  segments=sum(len(lab._runs(flag[:, b])) for b in range(flag.shape[1])),
                  gap_p95_mm=1000*quantile(features["min_dist"][flag], .95),
                  patch_gap_p95_mm=1000*quantile(patch[flag], .95),
                  positive_pairs_gap_over20mm=int((flag & (features["min_dist"] > .02)).sum()),
                  positive_pairs_gap_over50mm=int((flag & (features["min_dist"] > .05)).sum()),
                  patch_pairs_gap_over50mm=int((flag & (patch > .05)).sum()),
                  contact_target_shift_p95_mm=1000*quantile(delta[flag], .95),
                  reference_anchor_surface_error_max_mm=1000*float(old_surface.max()) if old_surface.size else 0.,
                  target_surface_error_max_mm=1000*float(projection_error.max()) if projection_error.size else 0.,
                  surface_changed_pairs=int((flag & features["surface_changed"]).sum()),
                  review_pairs=int(review.sum()), review_events=events,
                  motion_sha256=job["motion_sha256"], reference_contact_sha256=job["reference_sha256"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **labels)
    tmp.replace(out)
    report["refreshed_contact_sha256"] = sha(out)
    report_path = Path(job["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--simplified-collision", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, ref, out = (p.expanduser().resolve() for p in (args.dataset, args.reference_dataset, args.out))
    if out.exists():
        parser.error("Output already exists; snapshots are immutable")
    if out == root or out == ref or root in out.parents or ref in out.parents:
        parser.error("Output must be separate from the inputs")
    work = out.with_name(out.name+".building")
    meta = yaml.safe_load((root/"metadata.yaml").read_text())
    original = yaml.safe_load((ref/"metadata.yaml").read_text())
    ref_entries = {e["motion_file"]: e for e in original["motion_files"]}
    original_terrains = {e["terrain_id"]: e for e in original["terrains"]}
    terrains = {t["terrain_id"]: t for t in meta["terrains"]}
    files = {"metadata.yaml"}
    jobs = []
    for e in meta["motion_files"]:
        rel = e["motion_file"]
        if rel not in ref_entries:
            parser.error(f"Missing reference motion: {rel}")
        terrain = terrains[e["terrain_id"]]
        rt = original_terrains[ref_entries[rel]["terrain_id"]]
        if sha(root/terrain["terrain_file"]) != sha(ref/rt["terrain_file"]):
            parser.error(f"Reference/current terrain differ; explicit coordinate alignment is required: {rel}")
        files.add(rel)
        files.update(terrain[k] for k in ("terrain_file", "terrain_geoms") if terrain.get(k))
        side = Path(rel).with_suffix(".contacts.npz")
        jobs.append(dict(rel=rel, motion=str(root/rel), reference=str(ref/side), terrain=str(root/terrain["terrain_file"]),
                         out=str(work/side), report=str(work/"contact_refresh/"/Path(rel).with_suffix(".json")),
                         simplified_collision=args.simplified_collision, motion_sha256=sha(root/rel), reference_sha256=sha(ref/side)))
    spec_model = get_acp_g1_spec(args.simplified_collision).compile()
    manifest = dict(dataset=str(root), reference_dataset=str(ref), destination=str(out),
                    simplified_collision=args.simplified_collision,
                    code_sha256=sha(Path(__file__)), labeler_sha256=sha(Path(lab.__file__)),
                    asset_code_sha256=sha(Path(get_acp_g1_spec.__code__.co_filename)), asset_xml_sha256=sha(Path(ACP_G1_XML)),
                    compiled_model_sha256=hashlib.sha256(b"".join(np.asarray(getattr(spec_model,k)).tobytes() for k in
                        ("body_pos","body_quat","body_mass","body_inertia","geom_pos","geom_quat","geom_size","geom_type","jnt_range"))).hexdigest(),
                    files={rel:sha(root/rel) for rel in sorted(files)},
                    reference_contacts={j["rel"]:j["reference_sha256"] for j in jobs},
                    policy="Preserve original flags/timing; refresh geometry nearest original anchors; report contradictions")
    if work.exists():
        if not args.resume or not (work/"contact_refresh_manifest.json").exists():
            parser.error("Existing build requires --resume and its original manifest")
        if json.loads((work/"contact_refresh_manifest.json").read_text()) != manifest:
            parser.error("Input/code manifest changed; use another output")
    else:
        work.mkdir(parents=True)
        (work/"contact_refresh_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for rel in sorted(files):
        target=work/rel
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/rel,target)
    reports, pending = [], []
    for job in jobs:
        rp, sp = Path(job["report"]), Path(job["out"])
        if rp.exists() and sp.exists():
            row=json.loads(rp.read_text())
            if sha(sp) == row["refreshed_contact_sha256"]:
                reports.append(row)
                continue
        pending.append(job)
    print(f"Selected {len(jobs)} clips, {len(pending)} pending, workers={args.workers}",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(process,j) for j in pending]
        for future in as_completed(futures):
            row=future.result(); reports.append(row)
            print(f"{len(reports)}/{len(jobs)} {row['motion_file']} frames={row['frames']} review_pairs={row['review_pairs']}",flush=True)
    # Confirm assets did not change during the build and copy all authoritative checksums.
    for rel, expected in manifest["files"].items():
        if sha(root/rel)!=expected or sha(work/rel)!=expected:
            raise RuntimeError(f"Motion/terrain asset changed during refresh: {rel}")
    for j in jobs:
        if sha(Path(j["reference"]))!=j["reference_sha256"]:
            raise RuntimeError(f"Reference sidecar changed during refresh: {j['rel']}")
    reports.sort(key=lambda x:x["motion_file"])
    summary=dict(clips=len(reports),frames=sum(r["frames"] for r in reports),
                 positive_pairs=sum(r["positive_pairs"] for r in reports),changed_flag_pairs=0,
                 review_clips=sum(r["review_pairs"]>0 for r in reports),
                 review_pairs=sum(r["review_pairs"] for r in reports),
                 positive_pairs_gap_over50mm=sum(r["positive_pairs_gap_over50mm"] for r in reports),
                 target_surface_error_max_mm=max(r["target_surface_error_max_mm"] for r in reports),
                 notes=["Review thresholds never add/drop contact flags",
                        "Contact refresh does not repair hovering poses or prove dynamic feasibility",
                        "Original physics-derived timing is preserved; refreshed geometry is kinematic, not measured contact force"],
                 clip_results=reports)
    (work/"contact_refresh_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    write_csv(work/"contact_refresh.csv",reports)
    snapshot=dict(source=str(root),reference_dataset=str(ref),clips=len(jobs),terrains=len(meta["terrains"]),
                  changes=manifest["policy"],sha256={**manifest["files"],**{
                      str(Path(r["motion_file"]).with_suffix('.contacts.npz')):r["refreshed_contact_sha256"] for r in reports}})
    (work/"SNAPSHOT_MANIFEST.json").write_text(json.dumps(snapshot,indent=2)+"\n")
    work.rename(out)
    print(f"Published {out}: {summary['clips']} clips, flags unchanged, {summary['review_clips']} clips need review",flush=True)


if __name__ == "__main__":
    main()
