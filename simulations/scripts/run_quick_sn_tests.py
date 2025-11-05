import os
import argparse
import numpy as np
from yaml import safe_load as yload
from des_sn_hosts.simulations import aura

def single_z_counts(sim, z, n):
    df = sim._sample_SNe_z(z, n)
    return df

def multi_z_counts(sim, zarr, n_total):
    # power-law-ish z PDF
    w = np.array(zarr, float) ** 2.5
    w = w / w.sum()
    n_per_bin = sim._get_z_dist(w, n=n_total, zbins=zarr)
    sim.sample_SNe(zarr, n_per_bin, save_df=False)
    return sim.sim_df

def main():
    ap = argparse.ArgumentParser(description="Quick SN simulation tests")
    ap.add_argument("-c", "--config", required=True, help="Path to aura YAML config")
    ap.add_argument("-n", "--nsn", type=int, default=1000, help="Number of SNe per test")
    args = ap.parse_args()

    # Base sim using baked DTDs if present
    sim_baked = aura.Sim(args.config)
    zarr = np.sort(sim_baked.flux_df['z'].unique().astype(float))
    z0 = float(zarr[len(zarr)//2])

    # 1) 1000 SNe at a single redshift using baked DTD
    df1 = single_z_counts(sim_baked, z0, args.nsn)
    print(f"[Test 1] single-z baked: z={z0:.5f}, N={len(df1)}")

    # 2) 1000 SNe at a single redshift applying a different DTD (force recompute)
    sim_recomp = aura.Sim(args.config)
    sim_recomp.config['force_recompute_dtd'] = True
    sim_recomp.config['DTD'] = {'model': 'power_law', 'params': {'beta': 1.20, 'norm': 2.08e-13}}
    df2 = single_z_counts(sim_recomp, z0, args.nsn)
    print(f"[Test 2] single-z recompute DTD(beta=1.20): z={z0:.5f}, N={len(df2)}")

    # 3) 1000 SNe over a range of redshifts using baked DTD
    sim_baked2 = aura.Sim(args.config)
    df3 = multi_z_counts(sim_baked2, zarr, args.nsn)
    print(f"[Test 3] multi-z baked: N={len(df3)}, z-range=({zarr.min():.3f},{zarr.max():.3f})")

    # 4) 1000 SNe over a range of redshifts using a different DTD (force recompute)
    sim_recomp2 = aura.Sim(args.config)
    sim_recomp2.config['force_recompute_dtd'] = True
    sim_recomp2.config['DTD'] = {'model': 'power_law', 'params': {'beta': 1.05, 'norm': 2.08e-13}}
    df4 = multi_z_counts(sim_recomp2, zarr, args.nsn)
    print(f"[Test 4] multi-z recompute DTD(beta=1.05): N={len(df4)}, z-range=({zarr.min():.3f},{zarr.max():.3f})")

if __name__ == "__main__":
    main()