#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
from yaml import safe_load as yload
from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.dtd import compute_age_dist

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser(description="Simulate SNe using aura.Sim (single-process)")
    ap.add_argument('-c', '--config', required=True, help='aura config YAML path')
    ap.add_argument('-n', '--n-samples', type=int, default=None, help='Total SN samples (overrides config)')
    ap.add_argument('--out', default='', help='Output .h5 path (defaults from config)')
    ap.add_argument('--recompute-dtd', action='store_true', help='Recompute and bake DTD on-the-fly before simulation')
    ap.add_argument('--dtd-model', default='power_law', help='DTD model if recomputing')
    ap.add_argument('--dtd-params', default='', help='DTD params as k=v,k=v')
    ap.add_argument('--frac-low-z', type=float, default=0.0, help='Optional low-z fraction boost for sampling')
    args = ap.parse_args()

    sim = aura.Sim(args.config)

    # Resolve total samples
    n_total = args.n_samples
    if n_total is None:
        n_total = sim.config.get('n_samples', sim.config.get('simulate', {}).get('n_samples', None))
    if n_total is None:
        raise ValueError("Total n-samples not provided. Pass -n or set n_samples in config.")
    n_total = int(n_total)

    # Build per-z allocation from sim's redshift distribution
    n_samples_arr = sim.get_redshift_sample_counts(n_total, frac_low_z=args.frac_low_z)

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(sim.root_dir, 'sims', 'SNe', f"{sim.save_string}_SN_sim.h5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logger.info(f"Simulating {n_total} SNe across {len(sim.zarr)} z-bins -> {out_path}")

    if args.recompute_dtd:
        dtd_params = {}
        if args.dtd_params:
            for kv in args.dtd_params.split(','):
                if not kv:
                    continue
                k, v = kv.split('=')
                try:
                    dtd_params[k] = float(v)
                except ValueError:
                    dtd_params[k] = v
        try:
            if hasattr(sim, 'multi_df') and {'SFH_ages','SFH_m_formed'}.issubset(sim.multi_df.columns):
                sn_ages_col, sn_probs_col, pred_rates = [], [], []
                for a, m in zip(sim.multi_df['SFH_ages'], sim.multi_df['SFH_m_formed']):
                    a = np.asarray(a, float)
                    m = np.asarray(m, float)
                    dtd = compute_age_dist(a, model=args.dtd_model, **dtd_params)
                    dist = m * dtd
                    s = float(np.nansum(dist))
                    probs = dist / s if s and np.isfinite(s) else np.zeros_like(a)
                    sn_ages_col.append(a)
                    sn_probs_col.append(probs)
                    pred_rates.append(s)
                sim.multi_df['SN_ages'] = sn_ages_col
                sim.multi_df['SN_age_dist'] = sn_probs_col
                sim.multi_df['pred_rate_total'] = pred_rates
                sim.config['force_recompute_dtd'] = True
                sim.config['DTD'] = {'model': args.dtd_model, 'params': dtd_params}
                logger.info(f"Recomputed DTD baked (model={args.dtd_model}).")
            else:
                logger.warning("Cannot recompute DTD: SFH arrays not present in hostlib.")
        except Exception as e:
            logger.error(f"Failed DTD recompute: {e}")

    sim.sample_SNe(sim.zarr, n_samples_arr, savepath=out_path)
    logger.info("Done.")

if __name__ == '__main__':
    main()
