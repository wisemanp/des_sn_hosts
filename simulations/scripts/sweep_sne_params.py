#!/usr/bin/env python3
import os
import copy
import json
import argparse
import logging
import itertools
import numpy as np
from yaml import safe_load as yload, dump as ydump
from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.gal_functions import make_z_pdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def product_dict(d):
    keys = list(d.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in d.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def tag_from_params(p):
    parts = []
    for k, v in sorted(p.items()):
        if isinstance(v, float):
            parts.append(f"{k}_{v:.3f}")
        else:
            parts.append(f"{k}_{v}")
    return "__".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Sweep SNe simulation parameters and run aura.Sim for each combo")
    ap.add_argument('-c', '--config', required=True, help='Base aura config YAML')
    ap.add_argument('-g', '--grid', required=True, help='YAML with parameter grid under key grid:')
    ap.add_argument('-n', '--n-samples', type=int, required=True, help='Total SN samples per run')
    ap.add_argument('--out-dir', default='', help='Output directory; defaults to save.dir in config')
    ap.add_argument('--z-power', type=float, default=2.5, help='Exponent for z PDF')
    args = ap.parse_args()

    base_cfg = yload(open(args.config, 'r'))
    grid_cfg = yload(open(args.grid, 'r'))
    grid = grid_cfg.get('grid', {})
    if not grid:
        raise ValueError("Grid YAML must provide 'grid' mapping of parameters to lists")

    out_root = args.out_dir or base_cfg.get('save', {}).get('dir', os.path.join(os.getcwd(), 'sims'))
    os.makedirs(out_root, exist_ok=True)

    for params in product_dict(grid):
        # Build a tagged copy of the config with overrides
        cfg = copy.deepcopy(base_cfg)

        # Apply overrides into cfg: support dotted paths like SN_rv_model.params.sigma
        for k, v in params.items():
            path = k.split('.')
            ref = cfg
            for p in path[:-1]:
                if p not in ref:
                    ref[p] = {}
                ref = ref[p]
            ref[path[-1]] = v

        # Instantiate sim with this in-memory config
        # If aura.Sim expects a path, we can write a temp YAML; here we try allowing dict in constructor if supported
        sim = aura.Sim(cfg)

        zarr = np.sort(sim.flux_df['z'].unique().astype(float))
        pdf = make_z_pdf(zarr, power=args.z_power)
        n_samples_arr = sim._get_z_dist(pdf, n=args.n_samples, frac_low_z=0.0, zbins=zarr)

        tag = tag_from_params(params)
        base_name = sim.save_string if hasattr(sim, 'save_string') else 'sne_sim'
        out_path = os.path.join(out_root, f"{base_name}__{tag}__SN_sim.h5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        logger.info(f"Running sim: {tag} -> {out_path}")
        sim.sample_SNe(zarr, n_samples_arr, savepath=out_path)

        # Save a small manifest JSON of the overrides next to output
        with open(out_path.replace('.h5', '__params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    logger.info("All simulations completed.")


if __name__ == '__main__':
    main()
