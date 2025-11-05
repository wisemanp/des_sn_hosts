#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import pandas as pd
from yaml import safe_load as yload

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Build an SFH catalog (no spectra, no DTD)")
    ap.add_argument('-c', '--config', required=True, help='YAML config with input_sfh_path and save.dir')
    ap.add_argument('--time-res', type=int, default=5, help='Stride over SFH keys (e.g., 5)')
    ap.add_argument('--out', default='', help='Optional explicit output path (.h5)')
    args = ap.parse_args()

    cfg = yload(open(args.config, 'r'))
    in_path = cfg['input_sfh_path']
    out_dir = cfg['save']['dir']
    os.makedirs(out_dir, exist_ok=True)

    store = pd.HDFStore(in_path, 'r')
    keys = sorted([int(k.strip('/')) for k in store.keys()])

    rows = []
    for tf in keys[::-1][::args.time_res]:
        sfh = store[f'/{tf}']
        # Keep minimal columns; assume columns include 'z','age','m_formed','m_tot'
        if not {'z','age','m_formed','m_tot'}.issubset(set(sfh.columns)):
            logger.error(f"SFH table /{tf} missing required columns")
            continue
        # Build per-galaxy entries at this tf; keep one per row of sfh for now
        age_gyr = sfh['age'].values / 1000.0
        m_formed = sfh['m_formed'].values.astype(float)
        m_tot = float(sfh['m_tot'].iloc[-1])
        z_last = float(sfh['z'].iloc[-1])
        rows.append({
            't_f': float(tf),
            'z': z_last,
            'ages_gyr': age_gyr,
            'm_formed': m_formed,
            'm_tot': m_tot,
        })

    out_path = args.out or os.path.join(out_dir, 'sfh_catalog.h5')
    df = pd.DataFrame(rows)
    df.to_hdf(out_path, key='main', mode='w')
    logger.info(f"Wrote SFH catalog to {out_path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
