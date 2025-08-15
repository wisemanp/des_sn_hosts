import os
import sys
import csv
import pickle
import numpy as np
import pandas as pd
from yaml import safe_load as yload
from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.gal_functions import make_z_pdf
from des_sn_hosts.simulations.utils.helpers import prep_df_for_BBC

def load_config(cpath):
    with open(cpath, 'r') as f:
        return yload(f)



if __name__ == '__main__':
    cpath = sys.argv[1]
    cfg = load_config(cpath)
    
    model_config = os.path.split(cpath)[-1]
    model_name = model_config.split('.')[0]

    sim = aura.Sim(cpath)
    with open(cpath, 'r') as f:
        c = yload(f)

    # Optionally update config parameters here if needed

    sim.config = c
    zarr = np.sort(sim.flux_df['z'].unique().astype(float))
    z_pdf = make_z_pdf(zarr, power=2.5)
    n_samples_arr = sim._get_z_dist(z_pdf, n=cfg['n_samples'], frac_low_z=0., zbins=zarr)

    save_dir = os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/', cfg['save']['dir'])
    os.makedirs(save_dir, exist_ok=True)

    save_filename = f"{model_name}_SN_sim.h5"
    save_path = os.path.join(save_dir, save_filename)

    sim.sample_SNe(zarr, n_samples_arr, savepath=save_path)
    sim.sim_df = sim.sim_df[(sim.sim_df['x1'] < 3) & (sim.sim_df['x1'] > -3) & (sim.sim_df['c'] > -0.3) &
                            (sim.sim_df['c'] < 0.3) & (sim.sim_df['x1_err'] < 1) &
                            (sim.sim_df['c_err'] < 0.1)]
    sim.sim_df = sim.sim_df[sim.sim_df['mB'] < 25]
    sim.sim_df = sim.sim_df[sim.sim_df['eff_mask'] == 1]

    # Save filtered dataframe
    filtered_filename = f"{model_name}_filtered_SN_sim.h5"
    filtered_path = os.path.join(save_dir, filtered_filename)
    sim.sim_df.to_hdf(filtered_path, key='sim')

    # Convert to BBC format
    df, cols = prep_df_for_BBC(sim.sim_df)
    bbc_filename = f"{model_name}.FITRES"
    bbc_path = os.path.join(save_dir, bbc_filename)
    df[cols].to_csv(bbc_path, index=False, sep=' ', quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

    # Save parameter map for reference
    param_map = {'model_name': model_name, 'config_path': pth}
    pickle.dump(param_map, open(os.path.join(save_dir, 'param_name_map.pkl'), 'wb'))

    print(f"Simulation complete. Results saved to {save_dir}")