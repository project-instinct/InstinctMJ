"""Fit the sphere approximation of the nine ACP contact bodies (``rl/acp/body_spheres.py``) and write ``rl/acp/body_spheres.npz``.

    uv run --no-sync python -m instinct_mj.scripts.acp.fit_body_spheres [--simplified] [--no-save]

Prints the inward (hull surface not covered by the spheres) and outward (spheres poking out of the geometry) errors per body in mm."""

from __future__ import annotations

import argparse

from instinct_mj.rl.acp import body_spheres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--simplified", action="store_true", help="fit the simplified_collision=True model (analytic capsules / spheres) instead of the authors' meshes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fits, reports = body_spheres.fit_all(simplified=args.simplified, seed=args.seed)
    print(f"{'link':24s} {'n':>3s} {'in_max':>7s} {'in_p99':>7s} {'in_mean':>8s} {'out_max':>8s} {'out_p99':>8s}   [mm]")
    for k, r in reports.items():
        print(f"{k:24s} {r['n']:3d} {r['in_max']*1e3:7.2f} {r['in_p99']*1e3:7.2f} {r['in_mean']*1e3:8.2f} {r['out_max']*1e3:8.2f} {r['out_p99']*1e3:8.2f}")
    print("total spheres:", sum(r["n"] for r in reports.values()))
    if not args.no_save:
        from instinct_mj.assets.acp_g1 import get_acp_g1_spec

        body_spheres.save(fits, args.simplified, body_spheres.signature(get_acp_g1_spec(simplified_collision=args.simplified).compile()))
        print("saved", body_spheres.cache_path(args.simplified))


if __name__ == "__main__":
    main()
