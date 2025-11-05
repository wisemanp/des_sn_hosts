#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import pandas as pd
from yaml import safe_load as yload
from des_sn_hosts.simulations.utils.dtd import compute_age_dist

from des_sn_hosts.simulations.spectral_utils import (
    load_spectrum, interpolate_SFH, interpolate_SFH_pegase,
)
from des_sn_hosts.simulations.synspec import SynSpec

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def run_build_hosts_from_sfh(cfg: dict, sfh_df: pd.DataFrame, *, no_dtd: bool = False,
                             dtd_model: str = 'power_law', dtd_params: dict | None = None,
                             out_dir: str | None = None, out_path: str | None = None) -> str:
    """Core logic to build a hostlib from an SFH catalog.

    Returns the output HDF5 path.
    """
    if dtd_params is None:
        dtd_params = {}
    if out_dir is None:
        out_dir = cfg['save']['dir']
    os.makedirs(out_dir, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Build hosts (photometry/colours) from an SFH catalog")
    ap.add_argument('-c', '--config', required=True, help='YAML config with templates, save.dir, etc.')
    ap.add_argument('--sfh', required=True, help='Path to SFH catalog .h5 (from build_sfh_catalog.py)')
    ap.add_argument('--bake', action='store_true', help='Deprecated: baking is now default when DTD is applied')
    ap.add_argument('--no-dtd', action='store_true', help='Do not apply DTD/bake SN age distributions')
    ap.add_argument('--dtd-model', default='power_law', help='DTD model: power_law | two_component | broken_power_law')
    ap.add_argument('--dtd-params', default='', help='DTD params as k=v,k=v (e.g., beta=1.14,norm=2.08e-13)')
    ap.add_argument('--out', default='', help='Optional explicit output path (.h5)')
    args = ap.parse_args()

    cfg = yload(open(args.config, 'r'))
    out_dir = cfg['save']['dir']
    os.makedirs(out_dir, exist_ok=True)

    sfh_df = pd.read_hdf(args.sfh, key='main')

    # Parse DTD params
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

    # Templates
    if cfg.get('templates', 'BC03').upper() == 'BC03':
        aura_dir = os.environ.get('AURA_DIR', '/media/data3/wiseman/des/AURA/')
        with open(os.path.join(aura_dir, 'bc03_logt_list.dat')) as f:
            bc03_logt_list = [x.strip() for x in f.readlines()]
        bc03_logt_float_array = np.array([float(x) for x in bc03_logt_list])
        bc03_dir = '/media/data1/childress/des/galaxy_sfh_fitting/bc03_ssp_templates/'
        template_obj_list = [load_spectrum(f"{bc03_dir}bc03_chabrier_z02_{t}.spec") for t in bc03_logt_list]
        syn = SynSpec(template_obj_list=template_obj_list, neb=cfg.get('neb', False))
        def interp_fn(sfh_sub_df, mtot):
            return interpolate_SFH(sfh_sub_df, mtot, bc03_logt_float_array)
    else:
        # PEGASE
        tmpl_fn = cfg.get('templates_fn', None) or '/media/data3/wiseman/des/AURA/PEGASE/templates.h5'
        templates = pd.read_hdf(tmpl_fn, key='main')
        syn = SynSpec(library='PEGASE', template_dir=os.path.dirname(tmpl_fn), neb=cfg.get('neb', False))
        def interp_fn(sfh_sub_df, mtot):
            return interpolate_SFH_pegase(sfh_sub_df, templates['time'], mtot, templates['m_star'])

    rows = []
    # Prepare Av grid
    if cfg.get('av_step_type', 'lin') == 'log':
        av_arr = np.logspace(cfg.get('av_lo', 0.0), cfg.get('av_hi', 1.0), cfg.get('n_av', 20))
    else:
        av_arr = np.linspace(cfg.get('av_lo', 0.0), cfg.get('av_hi', 1.0), cfg.get('n_av', 20))

    for i, row in sfh_df.iterrows():
        z = float(row['z'])
        tf = float(row['t_f'])
        ages_gyr = np.asarray(row['ages_gyr'])
        m_formed = np.asarray(row['m_formed'])
        m_tot = float(row['m_tot'])
        age_yr = ages_gyr[-1] * 1000.0

        # Minimal SFH df for interpolation and derived host props
        df = pd.DataFrame({'age': ages_gyr * 1000.0, 'm_formed': m_formed})
        coeffs = interp_fn(df, m_tot)
        # Approximate recent SFR over last 250 Myr
        recent_mask = (df['age'] >= (df['age'].max() - 250.0))
        sfr = np.sum(df.loc[recent_mask, 'm_formed']) / (250.0)  # Msun per Myr; units consistent with upstream
        ssfr = (sfr / m_tot) if m_tot > 0 else 0.0
        mean_age = float(np.average(df['age'].values, weights=df['m_formed'].values / max(m_tot, 1e-12)))

    # Loop over Av grid
        for Av in av_arr:
            # Simple mass-dependent Rv prior (optional; override via cfg if provided)
            Rv = float(cfg.get('Rv', 3.1))
            galid = f"z_{z:.5f}_tf_{tf:.1f}_Av_{Av:.3f}_Rv_{Rv:.3f}"
            U_R, fluxes, colours = syn.calculate_model_fluxes_pw(
                z, coeffs,
                dust={'Av': float(Av), 'Rv': Rv, 'delta': 'none', 'law': 'CCM89'},
                neb=cfg.get('neb', False), logU=cfg.get('logU', -2), mtot=m_tot, age=age_yr, specsavename=galid
            )
            obs_flux = list(fluxes.values())
            # unpack colours in stable order
            U = colours.get('U'); B = colours.get('B'); V = colours.get('V'); R = colours.get('R'); I = colours.get('I')
            sdssu = colours.get('sdssu'); sdssg = colours.get('sdssg'); sdssr = colours.get('sdssr'); sdssi = colours.get('sdssi'); sdssz = colours.get('sdssz')

            # Compute DTD-based SN age distribution by default (unless --no-dtd)
            if not args.no_dtd:
                try:
                    dtd = compute_age_dist(ages_gyr, model=args.dtd_model, **dtd_params)
                    sn_age_dist = m_formed * dtd
                    norm = float(np.nansum(sn_age_dist))
                    if norm > 0 and np.isfinite(norm):
                        sn_age_probs = (sn_age_dist / norm).astype(float)
                    else:
                        sn_age_probs = np.zeros_like(ages_gyr, dtype=float)
                    pred_rate_total = float(np.nansum(sn_age_dist))
                except Exception as e:
                    logger.error(f"DTD computation failed for z={z:.5f}, tf={tf:.1f}: {e}")
                    sn_age_probs = np.zeros_like(ages_gyr, dtype=float)
                    pred_rate_total = 0.0
            else:
                sn_age_probs = None
                pred_rate_total = None

            res = dict(
                z=z, t_f=tf, mass=m_tot, ssfr=ssfr, mean_age=mean_age,
                Av=float(Av), Rv=Rv, delta='None',
                U_R=U_R[0], m_g=obs_flux[0], m_r=obs_flux[1], m_i=obs_flux[2], m_z=obs_flux[3],
                U=U, B=B, V=V, R=R, I=I, sdssu=sdssu, sdssg=sdssg, sdssr=sdssr, sdssi=sdssi, sdssz=sdssz,
                galid_spec=galid,
                # Persist SFH arrays in HDF (object arrays are acceptable for our usage)
                SFH_ages=ages_gyr.astype(float), SFH_m_formed=m_formed.astype(float),
            )
            if not args.no_dtd:
                res['SN_ages'] = ages_gyr.astype(float)
                res['SN_age_dist'] = sn_age_probs
                res['pred_rate_total'] = pred_rate_total
            rows.append(res)

    host_df = pd.DataFrame(rows)
    # Encode DTD model in filename for clarity
    dtd_tag = f"_{args.dtd_model}" if not args.no_dtd else ""
    out_path_final = args.out or os.path.join(out_dir, f'hostlib_from_sfh{dtd_tag}.h5')
    host_df.to_hdf(out_path_final, key='main', mode='w')
    logger.info(f"Wrote hostlib to {out_path_final} ({len(host_df)} rows). DTD applied: {not args.no_dtd} ({args.dtd_model if not args.no_dtd else 'N/A'})")
    return


if __name__ == '__main__':
    main()
