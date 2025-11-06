import os
import argparse
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d

def build_eff_df(eff_dir, fields=None, years=None, mag_min=10.0, mag_max=39.0, mag_step=0.05, poly_deg=13):
    # Defaults match existing OZDES helper
    if fields is None:
        fields = ['C12', 'X12', 'S12', 'E12', 'C3', 'X3']
    if years is None:
        years = ['Y123', 'Y4', 'Y5']

    mags = np.arange(mag_min, mag_max + mag_step/2.0, mag_step)
    eff_df = pd.DataFrame(index=mags)
    eff_df.index.name = 'mag'

    for y in years:
        y_suffix = y.replace('Y', '')  # files use Y<token> but path part uses the token after underscore
        for f in fields:
            path = os.path.join(eff_dir, 'efficiencies', f'eff_{f}_Y{y_suffix}.dat')
            if not os.path.exists(path):
                # Try alternate naming (already includes Y prefix)
                path = os.path.join(eff_dir, 'efficiencies', f'eff_{f}_{y}.dat')
            if not os.path.exists(path):
                # Skip if file truly missing
                continue

            eff = pd.read_csv(path, sep=' ', skipinitialspace=True)
            # Robust polynomial fit to smooth the curve
            coefs = poly.polyfit(eff['r_obs'], eff['HOSTEFF'], poly_deg)

            # Build a completeness curve limited to [10, 39] and monotonic caps
            slope_start = eff['r_obs'].loc[eff.sort_values('r_obs', ascending=False)['HOSTEFF'].idxmax()]
            slope_end = eff['r_obs'].loc[eff['HOSTEFF'].idxmin()]

            x_core = np.linspace(slope_start, slope_end, 1000)
            ffit = poly.polyval(x_core, coefs)

            # Extend to full mag range with caps
            x_left = np.linspace(mag_min, slope_start, 50)
            y_left = np.ones_like(x_left)
            x_right = np.linspace(slope_end, mag_max, 50)
            y_right = np.zeros_like(x_right)

            x_full = np.concatenate([x_left, x_core, x_right])
            y_full = np.concatenate([y_left, ffit, y_right])

            comp_fn = interp1d(x_full, y_full, kind='linear', bounds_error=False,
                               fill_value=(y_full[0], y_full[-1]))
            vals = np.clip(comp_fn(mags), 0.0, 1.0)

            col = f'{f}_{y}'
            eff_df[col] = vals

    return eff_df

def main():
    ap = argparse.ArgumentParser(description="Build and save full per-field detection efficiency table (no averaging).")
    ap.add_argument("--eff-dir", required=True, help="Directory containing efficiencies/eff_<FIELD>_Y<YEAR>.dat files.")
    ap.add_argument("--out", required=True, help="Output HDF5 path (will write key='eff').")
    ap.add_argument("--fields", default="C12,X12,S12,E12,C3,X3",
                    help="Comma-separated field groups to include (default: C12,X12,S12,E12,C3,X3).")
    ap.add_argument("--years", default="Y123,Y4,Y5", help="Comma-separated year groups (default: Y123,Y4,Y5).")
    ap.add_argument("--mag-min", type=float, default=10.0)
    ap.add_argument("--mag-max", type=float, default=39.0)
    ap.add_argument("--mag-step", type=float, default=0.05)
    ap.add_argument("--poly-deg", type=int, default=13, help="Polynomial degree for smoothing (default: 13).")
    args = ap.parse_args()

    fields = [s.strip() for s in args.fields.split(",") if s.strip()]
    years = [s.strip() for s in args.years.split(",") if s.strip()]

    eff_df = build_eff_df(
        eff_dir=args.eff_dir,
        fields=fields,
        years=years,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        mag_step=args.mag_step,
        poly_deg=args.poly_deg,
    )

    # Save full table (no averaging)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    eff_df.to_hdf(args.out, key='eff', mode='w', format='table', data_columns=['mag'])
    print(f"Wrote full efficiency table to {args.out} with key='eff' and columns: {list(eff_df.columns)}")

if __name__ == "__main__":
    main()