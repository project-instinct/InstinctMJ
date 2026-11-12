"""Collect omniretargeting output into an ACP TerrainMotion dataset for the reference player.

    uv run --no-sync python -m instinct_mj.scripts.acp.build_parkour_view_dataset \
        --out ~/Datasets/acp/hhtools_parkour_20261112 \
        --run ~/DataHHD/omniretargeting/out/hhtools_parkour_20261112

Every solved clip of a retarget run (``final_retargeted.npz`` plus the
``static_scene.obj`` it was solved against) becomes one entry of ``metadata.yaml``:
the motion is symlinked as ``<clip>.npz``, the terrain as ``terrains/<clip>.obj``,
and ``label_contacts`` can then drop ``<clip>.contacts.npz`` next to it.  The
player only reads this directory, so re-running the script is enough to publish
new clips; ``metadata.yaml`` gets a fresh mtime and the running server reloads.

Clip names come from the run's ``source.json`` (the source motion stem, e.g.
``D0429_PKO001_长方形越障_低_01_Sk00_000``), which keeps the PARC clip identity
instead of the hashed job directory name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

RUN_CLIP = "final_retargeted.npz"
RUN_SCENE = "static_scene.obj"
FLAT_GROUND = "terrains/_flat_ground.obj"


def clip_name(clip_dir: Path, motion: Path) -> str:
    source = clip_dir / "source.json"
    if source.exists():
        try:
            stem = Path(json.loads(source.read_text())["motion"]).stem
        except (OSError, ValueError, KeyError):
            stem = ""
        if stem:
            return stem
    # ``<clip>-<clip>.pkl-<hash>``: strip the duplicated job suffix.
    match = re.match(r"(.+?)-\1\.pkl-", motion.parent.name)
    return match.group(1) if match else motion.parent.name.split(".")[0]


def group_of(name: str) -> str:
    """PARC family, e.g. ``PKO001_长方形越障_低`` or ``前空翻``."""
    return re.sub(r"_\d+_Sk00_\d+$", "", name).removeprefix("D0429_").strip("_")


def flat_ground(root: Path, extent: float = 12.0) -> Path:
    path = root / FLAT_GROUND
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        corners = [(-extent, -extent, 0.0), (extent, -extent, 0.0),
                   (extent, extent, 0.0), (-extent, extent, 0.0)]
        path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in corners)
                        + "f 1 2 3\nf 1 3 4\n")
    return path


def collect(runs: list[Path]) -> dict[str, dict]:
    clips: dict[str, dict] = {}
    for run in runs:
        for motion in sorted((run / "clips").glob(f"*/{RUN_CLIP}")):
            clip_dir = motion.parent
            name = clip_name(clip_dir, motion)
            scene = clip_dir / RUN_SCENE
            mtime = motion.stat().st_mtime_ns
            previous = clips.get(name)
            if previous is not None and previous["mtime"] >= mtime:
                continue
            clips[name] = {"motion": motion, "scene": scene, "mtime": mtime, "run": run.name}
    return clips


def build(out: Path, clips: dict[str, dict], label: str, dry_run: bool) -> None:
    (out / "terrains").mkdir(parents=True, exist_ok=True)
    ground = flat_ground(out)
    entries, terrains, missing = [], [], []
    for terrain_id, (name, clip) in enumerate(sorted(clips.items())):
        scene = clip["scene"] if clip["scene"].exists() else ground
        if scene == ground and not clip["scene"].exists():
            missing.append(name)
        motion_link = out / f"{name}.npz"
        terrain_link = out / "terrains" / f"{name}.obj"
        if not dry_run:
            for link, target in ((motion_link, clip["motion"]), (terrain_link, scene)):
                if link.is_symlink() or link.exists():
                    if link.resolve() == target.resolve():
                        continue
                    link.unlink()
                os.symlink(target.resolve(), link)
        entries.append({"motion_file": f"{name}.npz", "terrain_id": terrain_id,
                        "group": group_of(name), "source_note": f"omniretargeting {clip['run']}"})
        terrains.append({"terrain_id": terrain_id, "terrain_file": f"terrains/{name}.obj"})
    metadata = {"label": label, "base_link": "pelvis",
                "motion_files": entries, "terrains": terrains}
    if not dry_run:
        text = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
        tmp = out / ".metadata.yaml.tmp"
        tmp.write_text(text)
        os.replace(tmp, out / "metadata.yaml")
    print(f"{len(entries)} clips -> {out}")
    if missing:
        print(f"{len(missing)} clips had no terrain, using the flat ground: "
              + ", ".join(missing))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="dataset folder the player reads")
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="omniretargeting run directory (repeatable)")
    parser.add_argument("--label", default="hhtools parkour (omniretargeting -> G1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    clips = collect([run.expanduser().resolve() for run in args.run])
    if not clips:
        raise SystemExit("no solved clips found")
    build(args.out.expanduser().resolve(), clips, args.label, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
