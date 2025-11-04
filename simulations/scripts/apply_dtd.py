import os
import argparse
import logging
import numpy as np
import pandas as pd
from des_sn_hosts.simulations.utils.dtd import compute_age_dist

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def _safe_norm(v):
    s = np.nansum(v)
    return v / s if np.isfinite(s) and s > 0 else np.zeros_like(v)

def main():
    ap = argparse.ArgumentParser(description="Apply a DTD to a base hostlib")
    ap.add_argument("--hostlib", required=True, help="Path to base hostlib HDF5 (no DTD baked)")
    ap.add_argument("--sfh_dir", required=True, help="Directory with per-host SFH .npz files")
    ap.add_argument("--model", default="power_law", help="DTD model name")
    ap.add_argument("--params", default="", help="DTD params as k=v,k=v (e.g., beta=1.14,norm=2.08e-13)")
    ap.add_argument("--out", required=True, help="Output HDF5 path for DTD-tagged hostlib")
    ap.add_argument("--snana_dat_dir", default="", help="If set, also write SN_ages .dat files here")
    args = ap.parse_args()

    params = {}
    if args.params:
        for kv in args.params.split(","):
            k, v = kv.split("=")
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v

    df = pd.read_hdf(args.hostlib, key="main")
    need_sfh = 'sfh_key' in df.columns
    if not need_sfh:
        raise RuntimeError("Base hostlib missing 'sfh_key' column; add SFH references during host build.")

    SN_ages_col = []
    SN_age_dist_col = []
    pred_rate_total_col = []

    if args.snana_dat_dir:
        os.makedirs(args.snana_dat_dir, exist_ok=True)

    for i, row in df.iterrows():
        sfh_path = os.path.join(args.sfh_dir, row['sfh_key'])
        dat = np.load(sfh_path)
        ages_gyr = dat['ages_gyr']
        m_formed = dat['m_formed']

        dtd = compute_age_dist(ages_gyr, model=args.model, **params)
        sn_age_dist = m_formed * dtd
        pred_rate_total = float(np.nansum(sn_age_dist))

        SN_ages_col.append(ages_gyr.astype(float))
        SN_age_dist_col.append(_safe_norm(sn_age_dist).astype(float))
        pred_rate_total_col.append(pred_rate_total)

        if args.snana_dat_dir:
            out_fn = f"{args.model}_z_{row['z']:.5f}_rv{row['Rv']:.2f}_{row['t_f']:.1f}_combined.dat"
            arr = np.column_stack([ages_gyr, _safe_norm(sn_age_dist)])
            np.savetxt(os.path.join(args.snana_dat_dir, out_fn), arr)

    df = df.copy()
    df['SN_ages'] = SN_ages_col       # object arrays; pandas stores them fine with fixed HDF
    df['SN_age_dist'] = SN_age_dist_col
    df['pred_rate_total'] = pred_rate_total_col
    df.to_hdf(args.out, key="main", mode="w")
    logger.info(f"Wrote DTD-tagged hostlib to {args.out}")

if __name__ == "__main__":
    main()