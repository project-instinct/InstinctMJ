"""Fixed-camera collision-geometry views for accepted local penetration repairs."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from instinct_mj.assets.acp_g1 import get_acp_g1_spec


def generate_gallery(source, output, manifest_path, gallery_dir, report_path):
    source, output, gallery = Path(source), Path(output), Path(gallery_dir)
    gallery.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(manifest_path).read_text())["clips"]
    meta = yaml.safe_load((source / "metadata.yaml").read_text())
    entry = {e["motion_file"]: e for e in meta["motion_files"]}
    terrains = {t["terrain_id"]: t for t in meta["terrains"]}
    rendered = []
    for rel, info in rows.items():
        if info.get("status") != "repaired":
            continue
        before = dict(np.load(source / rel))
        after = dict(np.load(output / rel))
        if hashlib.sha256((output / rel).read_bytes()).hexdigest() != info["output_sha"]:
            raise RuntimeError("output changed since acceptance: " + rel)
        spec = get_acp_g1_spec(simplified_collision=True)
        terrain = terrains[entry[rel]["terrain_id"]]
        for i, g in enumerate(yaml.safe_load((source / terrain["terrain_geoms"]).read_text())):
            spec.worldbody.add_geom(
                name=f"pen_terrain_{i}",
                type=getattr(mujoco.mjtGeom, "mjGEOM_" + g["type"].upper()),
                pos=g["pos"],
                quat=g.get("quat", [1, 0, 0, 0]),
                size=list(g["size"]) + [0.0] * (3 - len(g["size"])),
                rgba=[0.64, 0.67, 0.70, 1.0],
                group=0,
                contype=1,
                conaffinity=1,
            )
        model = spec.compile()
        data = mujoco.MjData(model)
        model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
        robot_ids = [g for g in range(model.ngeom) if model.geom_group[g] == 3]
        model.geom_rgba[robot_ids] = [0.18, 0.47, 0.76, 1.0]
        worst = min(int(info.get("worst_frame", 0)), len(before["base_pos_w"]) - 1)
        focus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, info.get("worst_geom", ""))
        if focus >= 0:
            model.geom_rgba[focus] = [0.9, 0.18, 0.10, 1.0]
        names = list(before["joint_names"])
        joints = [(model.jnt_qposadr[j], names.index(model.joint(j).name)) for j in range(1, model.njnt)]

        def pose(motion, frame):
            data.qpos[:3] = motion["base_pos_w"][frame]
            data.qpos[3:7] = motion["base_quat_w"][frame]
            for addr, col in joints:
                data.qpos[addr] = motion["joint_pos"][frame, col]
            mujoco.mj_forward(model, data)

        pose(before, worst)
        root_rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
        azimuth = float(np.degrees(np.arctan2(root_rotation[1, 0], root_rotation[0, 0])) + 90)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = 1
        option.geomgroup[3] = 1
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = azimuth
        camera.elevation = -12
        width, height = 640, 430
        frames = [max(0, worst - 4), worst, min(len(before["base_pos_w"]) - 1, worst + 4)]
        panels = [("overview", worst)] + [("closeup", f) for f in frames]
        if info.get("source_jumps"):
            jump = max(info["source_jumps"], key=lambda e: e["delta"])
            panels.append(("source joint jump", min(jump["frame"] + 1, len(before["base_pos_w"]) - 1)))
        canvas = Image.new("RGB", (2 * width, len(panels) * (height + 32)), "white")
        draw = ImageDraw.Draw(canvas)
        # Raise the model framebuffer height before creating the EGL renderer.
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            for row, (view, frame) in enumerate(panels):
                pose(before, frame)
                lookat = (
                    data.geom_xpos[focus].copy()
                    if focus >= 0 and view == "closeup"
                    else before["base_pos_w"][frame].copy()
                )
                camera.lookat[:] = lookat
                camera.distance = 1.1 if view == "closeup" else 3.1
                for col, motion in enumerate((before, after)):
                    pose(motion, frame)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render().copy()), (col * width, row * (height + 32) + 32))
                    label = ("SOURCE" if col == 0 else "REPAIRED") + f" | {view} | frame {frame}"
                    draw.text((col * width + 12, row * (height + 32) + 8), label, fill="black")
        name = Path(rel).stem + "_" + hashlib.sha1(rel.encode()).hexdigest()[:8] + ".jpg"
        canvas.save(gallery / name, quality=92)
        rendered.append(
            dict(
                clip=rel,
                image=name,
                **{
                    k: info.get(k)
                    for k in ("min_sd_before", "min_sd_after", "joint_max_deg", "root_max_cm", "body_max_cm")
                },
            )
        )
        print("gallery:", rel, flush=True)
    cards = []
    for r in rendered:
        cards.append(f"""<article data-name="{html.escape(r['clip'])}"><h2>{html.escape(r['clip'])}</h2>
<p>最小间隙 {1000*r['min_sd_before']:.3f} → {1000*r['min_sd_after']:.3f} mm；关节最大改动 {r['joint_max_deg']:.2f}°；根 {r['root_max_cm']:.2f} cm；body {r['body_max_cm']:.2f} cm。</p>
<a href="{r['image']}"><img loading="lazy" src="{r['image']}" alt="修前与修后对比"></a></article>""")
    unresolved = [rel for rel, r in rows.items() if r.get("status") == "unresolved"]
    doc = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v3_train11 穿模局部修复对比</title><style>body{font:16px/1.6 system-ui;max-width:1320px;margin:30px auto;padding:0 18px;color:#1d2630;background:#f4f6f8}article{background:white;margin:28px 0;padding:20px;border-radius:10px}h2{font-size:17px;overflow-wrap:anywhere}img{width:100%;height:auto}input{width:95%;padding:12px;font-size:16px}small{color:#536475}</style>
<h1>v3_train11：穿模局部修复</h1>"""
    deferred = sum(r.get("status") == "deferred_sparse" for r in rows.values())
    sparse_note = (
        f"sparse 组尚有 {deferred} 段等待新源数据。" if deferred else "sparse 组已按合入踝关节修复后的新源数据处理。"
    )
    doc += f"<p>已通过验收并生成对比图：{len(rendered)} 段。未通过并保留源动作：{len(unresolved)} 段。{sparse_note}</p>"
    doc += "<p>左：原动作；右：修后。红色是源数据间隙最小的碰撞体，蓝色是任务实际 v3 碰撞几何。相机与帧号完全一致；第一行全身侧视，其后三行放大检查部位。有源关节跳变的片段额外显示跳变帧全身图。</p><small>处理穿模及明确授权的源关节跳变；不进行贴面、钉脚、全局去抖或动作重定时。静态关键帧不能代替完整动作回放。</small>"
    if unresolved:
        doc += (
            "<details><summary>未通过片段（未采用候选）</summary><ul>"
            + "".join("<li>" + html.escape(x) + "</li>" for x in unresolved)
            + "</ul></details>"
        )
    doc += '<p><input id="filter" placeholder="按片段名筛选"></p>' + "\n".join(cards)
    doc += """<script>document.getElementById('filter').oninput=e=>{for(const a of document.querySelectorAll('article'))a.hidden=!a.dataset.name.toLowerCase().includes(e.target.value.toLowerCase())}</script></html>"""
    (gallery / "index.html").write_text(doc)
    Path(report_path).write_text(json.dumps(dict(rendered=rendered, unresolved=unresolved), indent=2))
