#!/usr/bin/env python3
import os
import csv
import argparse
import logging
import numpy as np
from yaml import safe_load as yload

from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.helpers import prep_df_for_BBC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser(description="Simulate SNe using aura.Sim (single-process) and save outputs for BBC.")
    ap.add_argument('-c', '--config', required=True, help='aura config YAML path')
    ap.add_argument('-n', '--n-samples', type=int, default=None, help='Total SN samples (overrides config)')
    args = ap.parse_args()

    cpath = args.config
    with open(cpath, 'r') as f:
        cfg = yload(f)

    # Derive model name from config filename
    model_config = os.path.split(cpath)[-1]
    model_name = os.path.splitext(model_config)[0]

    # Initialize sim
    sim = aura.Sim(cpath)

    # Resolve total samples: CLI overrides config
    n_total = args.n_samples
    if n_total is None:
        n_total = sim.config.get('n_samples', sim.config.get('simulate', {}).get('n_samples', None))
    if n_total is None:
        raise ValueError("Total n-samples not provided. Pass -n or set n_samples in the config.")
    n_total = int(n_total)

    # Build per-z allocation from sim's redshift distribution
    try:
        n_samples_arr = sim.get_redshift_sample_counts(n_total, frac_low_z=sim.config.get('frac_low_z', 0.0))
        zarr = sim.zarr
    except AttributeError:
        # Fallback: infer from flux_df if Sim doesn't yet expose redshift setup
        from des_sn_hosts.simulations.utils.gal_functions import make_z_pdf
        zarr = np.sort(sim.flux_df['z'].unique().astype(float))
        z_pdf = make_z_pdf(zarr, power=2.5)
        n_samples_arr = sim._get_z_dist(z_pdf, n=n_total, frac_low_z=0.0, zbins=zarr)

    # Output directories (for_BBC / from_BBC) and filenames (lifted from run_single_sim.py)
    save_dir = os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/', cfg['save']['dir'])
    os.makedirs(save_dir, exist_ok=True)
    receive_dir = os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/', cfg['save']['dir'])
    os.makedirs(receive_dir, exist_ok=True)

    # Unfiltered output
    save_filename = f"{model_name}_SN_sim.h5"
    save_path = os.path.join(save_dir, save_filename)

    logger.info(f"Simulating {n_total} SNe across {len(zarr)} z-bins -> {save_path}")
    sim.sample_SNe(zarr, n_samples_arr, savepath=save_path)

    # Apply same filters as run_single_sim.py
    sim.sim_df = sim.sim_df[(sim.sim_df['x1'] < 3) & (sim.sim_df['x1'] > -3) &
                            (sim.sim_df['c'] > -0.3) & (sim.sim_df['c'] < 0.3) &
                            (sim.sim_df['x1_err'] < 1) & (sim.sim_df['c_err'] < 0.1)]
    sim.sim_df = sim.sim_df[sim.sim_df['mB'] < 25]
    sim.sim_df = sim.sim_df[sim.sim_df['eff_mask'] == 1]

    # Save filtered dataframe
    filtered_filename = f"{model_name}_filtered_SN_sim.h5"
    filtered_path = os.path.join(save_dir, filtered_filename)
    sim.sim_df.to_hdf(filtered_path, key='sim')

    # Convert to BBC FITRES and save
    df_bbc, cols = prep_df_for_BBC(sim.sim_df)
    bbc_filename = f"{model_name}.FITRES"
    bbc_path = os.path.join(save_dir, bbc_filename)
    df_bbc[cols].to_csv(bbc_path, index=False, sep=' ', quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

    logger.info(f"Saved unfiltered: {save_path}")
    logger.info(f"Saved filtered:   {filtered_path}")
    logger.info(f"Saved FITRES:     {bbc_path}")
    logger.info(f"Receive dir ready at: {receive_dir}")
    logger.info("Done.")

if __name__ == '__main__':
    main()
