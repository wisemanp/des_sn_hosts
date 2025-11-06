import numpy as np
import pandas as pd
from astropy.table import Table
import os
from yaml import safe_load as yload
import scipy.stats as stats
from scipy.stats import norm
import sys
import pickle
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
import time
from .models.sn_model import SN_Model
from .utils.gal_functions import schechter, ozdes_efficiency, interpolate_zdf
from .utils.HR_functions import get_mu_res_step, get_mu_res_nostep, chisq_mu_res_nostep, chisq_mu_res_step,chisq_mu_res_nostep_old
import logging
from scipy.interpolate import interp1d
import fnmatch

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import itertools
from tqdm import tqdm


aura_dir = os.environ['AURA_DIR']
idx = pd.IndexSlice

age_grid = np.arange(0,13.7,0.0005)
age_grid_index = ['%.4f'%a for a in age_grid]

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more verbosity
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

class Sim(SN_Model):
    """
    Simulation class for drawing SN samples from a galaxy population.

    Accepts either a path to a YAML config file or an in-memory dict.
    """

    def __init__(self, conf, cosmo='default'):
        self.config = self._get_config(conf)

        root_dir = self.config['config']['root_dir']
        self.root_dir = os.environ.get(root_dir[1:]) if root_dir.startswith('$') else root_dir

        self.fig_dir = os.path.join(self.root_dir, 'figs/')
        self.eff_dir = self.config['config']['efficiency_dir']
        self.flux_df = self._load_flux_df(self.config['hostlib_fn'])

        self._calculate_absolute_rates()
        self._make_multi_index()

        self.cosmo = FlatLambdaCDM(70, 0.3) if cosmo == 'default' else cosmo
        self._get_funcs()

        # NEW: init efficiencies and fields once
        self._init_efficiency_lookup()
        self._init_fields()

    # NEW: one-time efficiency initialization (reads full eff_df; no averaging)
    def _init_efficiency_lookup(self):
        """
        Initialize efficiency interpolators from a precomputed HDF5 table (full eff_df),
        or fall back to ozdes_efficiency() as a last resort. Supports per-field column sets.
        Config examples:
          efficiency:
            table: /path/to/eff_table.h5
            key: eff            # optional, defaults to 'eff'
            patterns:           # optional glob patterns per field to select columns
              shallow: ["C12_*", "X12_*", "S12_*", "E12_*"]
              deep:    ["C3_*", "X3_*"]
          fields:
            shallow:
              prob: 0.7
              eff_columns: ["C12_Y123","X12_Y123"]   # optional explicit list overrides patterns
            deep:
              prob: 0.3
        """
        self.eff_lookup = {}            # column -> interp1d over mag
        self.eff_columns_by_field = {}  # field -> list of eff_df columns

        eff_cfg = self.config.get('efficiency', {})
        table_path = eff_cfg.get('table', None)
        key = eff_cfg.get('key', 'eff')

        if table_path and os.path.exists(table_path):
            try:
                eff_df = pd.read_hdf(table_path, key=key)
                # Use index if named 'mag', else a 'mag' column
                if 'mag' in eff_df.columns:
                    mags = eff_df['mag'].values.astype(float)
                    eff_df = eff_df.drop(columns=['mag'])
                else:
                    mags = eff_df.index.values.astype(float)

                # Build an interpolator per column (bounded to endpoints)
                for col in eff_df.columns:
                    y = np.asarray(eff_df[col].values, dtype=float)
                    # sanitize
                    y = np.clip(y, 0.0, 1.0)
                    self.eff_lookup[col] = interp1d(
                        mags, y, kind='linear', bounds_error=False,
                        fill_value=(y[0], y[-1])
                    )
                logger.info(f"Loaded efficiency table '{key}' from {table_path} with {len(self.eff_lookup)} curves.")

                # Map columns to fields using explicit lists or glob patterns
                patterns = eff_cfg.get('patterns', {})
                if self.config.get('fields'):
                    for fname, fcfg in self.config['fields'].items():
                        cols = fcfg.get('eff_columns', None)
                        if cols:
                            selected = [c for c in cols if c in self.eff_lookup]
                        else:
                            pats = patterns.get(fname, [])
                            matched = []
                            for pat in pats:
                                matched.extend(fnmatch.filter(list(self.eff_lookup.keys()), pat))
                            selected = sorted(set(matched))
                        # If nothing matched, default to all columns to avoid empty
                        if not selected:
                            selected = list(self.eff_lookup.keys())
                            logger.warning(f"No efficiency columns matched for field '{fname}'. Using all columns.")
                        self.eff_columns_by_field[fname] = selected
                else:
                    # Single global bucket uses all columns
                    self.eff_columns_by_field['global'] = list(self.eff_lookup.keys())

                return
            except Exception as e:
                logger.warning(f"Failed loading precomputed efficiency table: {e}. Falling back to ozdes_efficiency().")

        # Legacy fallback: single global OZDES efficiency (mean/std)
        try:
            mean_eff_func, std_eff_func = ozdes_efficiency(self.eff_dir)
            # Wrap them to look like our per-column interpolators (std kept separate via config if needed)
            self.eff_lookup['global_mean'] = mean_eff_func
            self.eff_columns_by_field['global'] = ['global_mean']
            logger.info("Initialized global OZDES efficiency (legacy).")
        except Exception as e:
            logger.error(f"Could not initialize detection efficiency: {e}")
            # Safe constant zero function
            self.eff_lookup['global_mean'] = lambda m: np.zeros_like(np.atleast_1d(m), float)
            self.eff_columns_by_field['global'] = ['global_mean']

    # NEW: simple field setup
    def _init_fields(self):
        """
        Configure fields and their selection probabilities and noise model.
        Example config:
          fields:
            shallow:
              prob: 0.7
              eff_field: shallow
              noise:
                mB_err_scale: 1.0
                c_err_floor: 0.02
                x1_err_floor: 0.08
            deep:
              prob: 0.3
              eff_field: deep
              noise:
                mB_err_scale: 0.9
                c_err_floor: 0.015
                x1_err_floor: 0.07
        """
        self.fields_cfg = self.config.get('fields', None)
        if self.fields_cfg:
            names = list(self.fields_cfg.keys())
            probs = np.array([self.fields_cfg[n].get('prob', 1.0) for n in names], float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
            self._field_names = names
            self._field_probs = probs
            logger.info(f"Initialized fields: {names} with probs {probs}.")
        else:
            self._field_names = ['global']
            self._field_probs = np.array([1.0])

    # helper to get proper eff funcs for a given field
    def _eff_fns_for_field(self, field_name):
        """
        Return list of eff column names assigned to this field.
        """
        if field_name in self.eff_columns_by_field:
            return self.eff_columns_by_field[field_name]
        if 'global' in self.eff_columns_by_field:
            return self.eff_columns_by_field['global']
        # Fallback: all available
        return list(self.eff_lookup.keys())

    # ----------------------
    # CONFIG & DATA LOADING
    # ----------------------
    def _get_config(self, conf):
        # conf can be a path (str) or a pre-loaded dict
        if isinstance(conf, dict):
            return conf
        if isinstance(conf, str):
            if not os.path.exists(conf):
                raise FileNotFoundError(f"Config path not found: {conf}")
            with open(conf, 'r') as f:
                return yload(f)
        raise TypeError("conf must be a dict or a path to a YAML config file")

    def _load_flux_df(self, fn):
        df = pd.read_hdf(fn)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                logger.warning(f"Non-numeric column in flux_df: {col}")
        return df

    # ----------------------
    # CALCULATIONS
    # ----------------------
    def _calculate_absolute_rates(self):
        self.flux_df['SF'] = ((np.log10(self.flux_df['ssfr'].values) > -10.0).astype(int) == 0).astype(int)
        self.flux_df['phi'] = self.flux_df[['z', 'mass', 'SF']].apply(
            lambda x: schechter(x[0], np.log10(x[1]), x[2]), axis=1
        )
        self.flux_df['N_x1_lo'] = self.flux_df['pred_rate_x1_lo'] * self.flux_df['phi']
        self.flux_df['N_x1_hi'] = self.flux_df['pred_rate_x1_hi'] * self.flux_df['phi']
        self.flux_df['N_total'] = self.flux_df['pred_rate_total'] * self.flux_df['phi']

    def _make_multi_index(self):
        z_str = self.flux_df['z'].apply(lambda x: f"{x:.5f}")
        mass_str = self.flux_df['mass'].apply(lambda x: f"{x:.2f}")
        Av_str = self.flux_df['Av'].apply(lambda x: f"{x:.5f}")
        self.multi_df = self.flux_df.set_index([z_str, mass_str, Av_str])

    def _get_funcs(self):
        self.rv_func = getattr(self, self.config['SN_rv_model']['model'])
        self.host_Av_func = getattr(self, self.config['Host_Av_model']['model'])
        self.E_func = getattr(self, self.config['SN_E_model']['model'])
        self.colour_func = getattr(self, self.config['SN_colour_model']['model'])
        self.x1_func = getattr(self, self.config['x1_model']['model'])
        self.mb_func = getattr(self, self.config['mB_model']['model'])
        self.save_string = "_".join([
            self.rv_func.__name__,
            self.host_Av_func.__name__,
            self.E_func.__name__,
            self.colour_func.__name__,
            self.x1_func.__name__,
            self.mb_func.__name__
        ])
    def _get_z_dist(self, zsource, n=1000, frac_low_z=0.0, zbins=None):
        """
        Get the number of SNe to draw in each redshift bin.

        Parameters
        ----------
        zsource : array-like
            Either:
            - A normalized PDF for zbins, OR
            - A continuous or discrete redshift sample to histogram.
        n : int
            Total number of SNe to simulate.
        frac_low_z : float
            Fraction of SNe to force into the lowest redshift bin.
        zbins : array-like
            Allowed redshift bin centers.

        Returns
        -------
        counts : np.ndarray
            Number of SNe per redshift bin, in same order as zbins.
        """
        if zbins is None:
            raise ValueError("zbins must be provided and match flux_df keys.")
        zbins = np.array(zbins, dtype=float)

        if len(zsource) == len(zbins) and np.isclose(np.sum(zsource), 1.0):
            # Assume zsource is already a normalized PDF
            pdf = np.array(zsource, dtype=float)
        elif not np.all(np.isin(zsource, zbins)):
            # Continuous input — bin it
            edges = np.concatenate([
                [zbins[0] - (zbins[1] - zbins[0]) / 2],
                (zbins[:-1] + zbins[1:]) / 2,
                [zbins[-1] + (zbins[-1] - zbins[-2]) / 2]
            ])
            hist, _ = np.histogram(zsource, bins=edges)
            pdf = hist.astype(float)
            pdf /= pdf.sum()
        else:
            # Discrete input of allowed bins
            unique, counts = np.unique(zsource, return_counts=True)
            pdf = np.array([counts[unique == z].sum() if z in unique else 0 for z in zbins], dtype=float)
            pdf /= pdf.sum()

        counts = np.random.multinomial(n, pdf)

        if frac_low_z > 0:
            low_z_count = int(np.round(frac_low_z * n))
            counts[0] += low_z_count
            # Remove from other bins proportionally
            if counts.sum() > n:
                excess = counts.sum() - n
                nonzero_idx = np.where(counts[1:] > 0)[0] + 1
                for i in nonzero_idx:
                    take = min(counts[i], int(round(excess * (counts[i] / counts[1:].sum()))))
                    counts[i] -= take
                counts[counts < 0] = 0

        return counts

    # ----------------------
    # MAIN SAMPLING
    # ----------------------
    def sample_SNe(self, z_arr, n_samples_arr, save_df=True, savepath='default'):
        self.sim_df = pd.DataFrame()
        for z, n in zip(z_arr, n_samples_arr):
            self.sim_df = pd.concat([self.sim_df, self._sample_SNe_z(z, n)])

        if save_df:
            if savepath == 'default':
                savepath = os.path.join(self.root_dir, 'sims', 'SNe', f"{self.save_string}_SN_sim.h5")
            self.savepath = savepath
            self.sim_df.to_hdf(self.savepath, key='sim')

    def _sample_SNe_z(self, z, n_samples):
        if n_samples == 0:
            logger.info(f"No SNe to sample at z={z:.5f}")
            return pd.DataFrame()

        rng = np.random.default_rng()
        args = {'n': int(n_samples), 'distmod': self.cosmo.distmod(z).value}

        # Get galaxies at this redshift
        logger.debug(f"Sampling SNe at z={z:.5f} with n_samples={n_samples}")
        z_df = self.multi_df.loc[f"{z:.5f}"].copy()
        z_df.replace({'N_total': {0.0: np.nan}}, inplace=True)
        z_df.dropna(subset=['N_total'], inplace=True)
        z_df['N_SN_float'] = z_df['N_total'] / z_df['N_total'].min()
        z_df['N_SN_int'] = z_df['N_SN_float'].astype(int)

        # Interpolate over masses for each Av
        marr = np.logspace(6, 11.6, 100)
        resampled_df = pd.DataFrame()
        for av in z_df.Av.unique():
            logger.debug(f"Interpolating Av={av:.5f}")
            av_df = z_df.loc[idx[:, f"{av:.5f}", :]]
            av_df = interpolate_zdf(av_df, marr)
            resampled_df = pd.concat([resampled_df, av_df])

        # Build new_zdf with empty age dists
        Av_str = resampled_df['Av'].apply(lambda x: f"{x:.5f}")
        mass_str = resampled_df['mass'].apply(lambda x: f"{x:.2f}")
        new_zdf = resampled_df.set_index([mass_str, Av_str])
        new_zdf['SN_ages'] = [age_grid.copy() for _ in range(len(new_zdf))]
        new_zdf['SN_age_dist'] = [np.zeros(len(age_grid)) for _ in range(len(new_zdf))]

        # Fill age distributions per mass bin
        for mass_bin, g in z_df.groupby(pd.cut(z_df['mass'], bins=marr)):
            logger.debug(f"Processing mass bin: {np.log10(mass_bin.mid)}")
            if len(g) == 0:
                #logger.info(f"Skipping empty mass bin: {np.log10(mass_bin.mid)}")
                continue
            min_av = g.Av.astype(float).min()
            g_Av_0 = g.loc[idx[:, f"{min_av:.5f}", :]]

            age_df = pd.DataFrame(index=age_grid_index)
            for k in g_Av_0.index.unique():
                sub_row = g_Av_0.loc[k]
                # Prefer baked columns if present in hostlib
                used = False
                try:
                    if isinstance(sub_row, pd.DataFrame):
                        sub_row0 = sub_row.iloc[0]
                    else:
                        sub_row0 = sub_row
                    # grab t_f early in case we need legacy .dat fallback
                    tf = None
                    try:
                        tf = float(sub_row0.get('t_f', np.nan))
                    except Exception:
                        tf = None
                    # Use baked SN ages unless recompute is explicitly requested
                    if (not self.config.get('force_recompute_dtd', False)) and \
                       ('SN_ages' in sub_row0 and 'SN_age_dist' in sub_row0):
                        ages_arr = np.asarray(sub_row0['SN_ages'], dtype=float)
                        dist_arr = np.asarray(sub_row0['SN_age_dist'], dtype=float)
                        if ages_arr.size and dist_arr.size and ages_arr.size == dist_arr.size:
                            dist_norm = np.nansum(dist_arr)
                            probs = (dist_arr / dist_norm) if dist_norm > 0 and np.isfinite(dist_norm) else np.zeros_like(dist_arr)
                            age_inds = [f"{a:.4f}" for a in ages_arr]
                            age_df.loc[age_inds, f"{float(k):.2f}"] = probs
                            used = True
                except Exception as e:
                    logger.debug(f"Baked SN-age not usable for index {k}: {e}")

                if used:
                    continue

                # Recompute from SFH arrays if available 
                try:
                    if 'SFH_ages' in sub_row0 and 'SFH_m_formed' in sub_row0:
                        sfh_ages = np.asarray(sub_row0['SFH_ages'], dtype=float)
                        sfh_m = np.asarray(sub_row0['SFH_m_formed'], dtype=float)
                        if sfh_ages.size and sfh_m.size and sfh_ages.size == sfh_m.size:
                            # DTD config
                            dtd_cfg = self.config.get('DTD', {'model': 'power_law', 'params': {'beta': 1.14, 'norm': 2.08e-13}})
                            dtd_model = dtd_cfg.get('model', 'power_law')
                            dtd_params = dtd_cfg.get('params', {})
                            try:
                                from des_sn_hosts.simulations.utils.dtd import compute_age_dist
                                dtd_vals = compute_age_dist(sfh_ages, model=dtd_model, **dtd_params)
                            except Exception as e:
                                logger.error(f"DTD compute failed (model={dtd_model}) for index {k}: {e}")
                                dtd_vals = np.zeros_like(sfh_ages)
                            dist = sfh_m * dtd_vals
                            # Map onto the simulation age grid
                            dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
                            try:
                                # age_grid is the target grid (in Gyr). Ensure monotonic for interp.
                                order = np.argsort(sfh_ages)
                                probs_grid = np.interp(age_grid, sfh_ages[order], dist[order], left=0.0, right=0.0)
                            except Exception:
                                # Fallback: if interpolation fails, try nearest via indexing trick
                                probs_grid = np.zeros_like(age_grid)
                            sum_probs = float(np.nansum(probs_grid))
                            probs_grid = (probs_grid / sum_probs) if sum_probs > 0 and np.isfinite(sum_probs) else np.zeros_like(probs_grid)
                            age_inds = [f"{a:.4f}" for a in age_grid]
                            age_df.loc[age_inds, f"{float(k):.2f}"] = probs_grid
                            used = True
                except Exception as e:
                    logger.warning(f"Could not recompute SN-age from SFH for index {k}: {e}")

                # Legacy fallback: read .dat files if neither baked nor SFH recompute worked
                if not used and tf is not None and np.isfinite(tf):
                    try:
                        hostlib_base = os.path.basename(self.config['hostlib_fn'])
                        split_z = hostlib_base.split('z')
                        split_rv = hostlib_base.split('rv')
                        ext = f"{split_z[0]}z_{z:.5f}_rv{split_rv[1][:-12]}_{tf:.1f}_combined.dat"
                        sn_dir = os.path.join(os.path.dirname(self.config['hostlib_fn']), 'SN_ages')
                        new_fn = os.path.join(sn_dir, ext)
                        logger.debug(f"Fallback to legacy .dat SN-age file: {new_fn}")
                        sub = pd.read_csv(new_fn, sep=' ', names=['SN_ages', 'SN_age_dist'])
                        age_inds = [f"{a:.4f}" for a in sub['SN_ages']]
                        vals = sub['SN_age_dist'].values.astype(float)
                        s = np.nansum(vals)
                        probs = (vals / s) if s > 0 and np.isfinite(s) else np.zeros_like(vals)
                        age_df.loc[age_inds, f"{float(k):.2f}"] = probs
                        used = True
                    except Exception as e:
                        logger.warning(f"Legacy .dat SN-age read failed for index {k}: {e}")

                if not used:
                    # Leave zeros; downstream nanmean over columns will handle it
                    logger.warning("No SN age distribution available; leaving zeros for this bin.")

            age_df.fillna(0, inplace=True)
            avg_dist = np.nanmean(age_df, axis=1)

            # Assign to all matching rows
            for idx_key in new_zdf.index[new_zdf['mass'].between(
                g.mass.min(), g.mass.max(), inclusive='both')]:
                new_zdf.at[idx_key, 'SN_age_dist'] = avg_dist.copy()

        # Select galaxies & sample ages
        m_rates_s = new_zdf.groupby(level=0, sort=False)['N_SN_int'].mean()
        m_inds = m_rates_s.index
        m_rates = m_rates_s.values
        logger.debug(f"Sampling galaxy masses: {m_inds}")
        m_samples = rng.choice(m_inds, p=m_rates / np.sum(m_rates), size=int(n_samples))
        m_av0_samples = [(m, f"{rng.choice(new_zdf.loc[m].Av.values):.5f}") for m in m_samples]

        sn_ages = []
        for m_av in m_av0_samples:
            probs = new_zdf.loc[m_av, 'SN_age_dist']
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
            sn_age = rng.choice(new_zdf.loc[m_av, 'SN_ages'], p=probs)
            sn_ages.append(sn_age)
            logger.debug(f"Sampled SN age for {m_av}: {sn_age}")

        # Continue with args for light curve parameters
        gals_df = new_zdf.loc[m_av0_samples]
        args['Av_grid'] = new_zdf.Av.unique()
        args['mass'] = gals_df.mass.values
        args['ssfr'] = gals_df.ssfr.values
        args['sfr'] = args['mass'] * args['ssfr']
        args['mean_ages'] = gals_df.mean_age.values
        args['SN_age'] = np.array(sn_ages)
        args['rv'] = self.rv_func(args, self.config['SN_rv_model']['params'])

        if self.config['SN_E_model']['model'] in ['E_calc', 'E_from_host_random']:
            args['host_Av'] = self.host_Av_func(args, self.config['Host_Av_model']['params'])
            args['E'] = self.E_func(args, self.config['SN_E_model']['params'])
        else:
            args['E'] = self.E_func(args, self.config['SN_E_model']['params'])
            args['host_Av'] = self.host_Av_func(args, self.config['Host_Av_model']['params'])
        m_av_samples_inds = [[m_samples[i],'%.5f'%(args['host_Av'][i])] for i in range(len(args['host_Av']))]
        gals_df = new_zdf.loc[m_av_samples_inds]  # ensure same length as other vectors
        # Add z per SN so brightness models can use it
        args['z'] = np.full(len(gals_df), float(z), dtype=float)

        # Restore per-SN colours/mags from the dust-attenuated selection
        # Needed for efficiency lookup (m_r) and for downstream analyses (U-R)
        if {'U','R'}.issubset(gals_df.columns):
            args['U-R'] = gals_df['U'].values - gals_df['R'].values
        for band in ['g', 'r', 'i', 'z']:
            if f"m_{band}" in gals_df.columns:
                args[f"m_{band}"] = gals_df[f"m_{band}"].values

        # --- Field assignment and efficiencies (using preloaded eff_df interpolators) ---
        # Assign a field label to each SN
        fields = np.random.choice(self._field_names, size=len(args['m_r']), p=self._field_probs)
        args['field'] = fields

        # Evaluate detection efficiencies per SN without averaging across fields
        effs = np.zeros_like(args['m_r'], dtype=float)
        chosen_eff_cols = np.empty(len(args['m_r']), dtype=object)

        for fname in np.unique(fields):
            sel = (fields == fname)
            cols = self._eff_fns_for_field(fname)  # list of eff_df columns for this field
            if not cols:
                cols = list(self.eff_lookup.keys())
            # Randomly choose one curve per SN in this field cohort
            rnd_idx = np.random.integers(0, len(cols)-1, size=sel.sum())
            for j, i_sn in enumerate(np.where(sel)[0]):
                col = cols[rnd_idx[j]]
                fn = self.eff_lookup[col]
                effs[i_sn] = np.clip(fn(args['m_r'][i_sn]), 0.0, 1.0)
                chosen_eff_cols[i_sn] = col

        args['eff_col'] = chosen_eff_cols.tolist()
        args['eff_mask'] = [np.random.choice([0, 1], p=[1 - effs[i], effs[i]]) for i in range(len(effs))]

        # Colours & magnitudes
        args = self.colour_func(args, self.config['SN_colour_model']['params'])
        args = self.x1_func(args, self.config['x1_model']['params'])
        args['mB'], args['alpha_SN'], args['beta_SN'] = self.mb_func(args, self.config['mB_model']['params'])

        # Errors and noise
        args['mB_err'] = [
            np.max([0.025, np.random.normal(10**(0.395*(mB-1.5) - 10.15) + 0.025,
                                            np.max([0.003, 0.003*(mB-20)]))])
            for mB in args['mB']
        ]
        args['c_err'] = [np.max([0.02, np.random.normal((0.78007*err + 0.00256), 0.003)])
                         for err in args['mB_err']]
        args['c_noise'] = stats.norm(0, args['c_err']).rvs(size=len(args['c']))
        if not bool(self.config.get('c_perfect', False)):
            args['c'] = args['c'] + args['c_noise']

        args['x1_err'] = [np.max([0.08, np.random.normal((11.525*err - 0.1075), 0.05)])
                          for err in args['mB_err']]
        args['x1_noise'] = stats.norm(0, args['x1_err']).rvs(size=len(args['x1']))
        args['x1_int'] = args['x1'].copy()
        args['x1'] = args['x1'] + args['x1_noise']
        args['cov_mB_x1'], args['cov_mB_c'], args['cov_x1_c'] = 0, 0, 0

        args['distmod'] = np.ones_like(args['c']) * args['distmod']
        del args['Av_grid']

        z_sim_df = pd.DataFrame(args)
        z_sim_df['z'] = z
        logger.info(f"Finished sampling SNe at z={z:.5f}, n={n_samples}")
        return z_sim_df

    def load_sim(self,path):
        self.sim_df = pd.read_hdf(path,key='sim')
    def fit_mu_res(self):
        self.fitter = getattr(self,self.config['mu_res_fitter']['fitter'])
        self.fitter(self.config['mu_res_fitter']['params'])
        self.getter = getattr(self,self.config['mu_res_fitter']['fitter'].replace('fit','get'))
        self.getter(self.res['x'],self.config['mu_res_fitter']['params'])
    def fit_mu_res_nostep(self,params):
        x0 =[0.1,3.1,-19.5]
        res =minimize(chisq_mu_res_nostep,x0,args=[self.sim_df,params,self.cosmo])
        self.alpha_fit,self.beta_fit,self.MB_fit = res['x'][0],res['x'][1],res['x'][2]
        self.res = res

    def fit_mu_res_nostep_old(self,params):
        x0 =[0.1,3.1,-19.5]
        res =minimize(chisq_mu_res_nostep_old,x0,args=[self.sim_df,params,self.cosmo])
        self.alpha_fit,self.beta_fit,self.MB_fit = res['x'][0],res['x'][1],res['x'][2]
        self.res = res

    def get_mu_res_nostep(self,res,params):
        self.sim_df['mu_res'] = get_mu_res_nostep(res,self.sim_df,params,self.cosmo)
        self.sim_df['mu_res_err'] = self.sim_df['mB_err']
    def get_mu_res_nostep_old(self,res,params):
        self.sim_df['mu_res'] = get_mu_res_nostep(res,self.sim_df,params,self.cosmo)
        self.sim_df['mu_res_err'] = self.sim_df['mB_err']

    def fit_mu_res_step(self,params):
        x0 =[0.1,3.1,-19.5]
        res =minimize(chisq_mu_res_step,x0,args=[self.sim_df,params,self.cosmo])
        self.alpha_fit,self.beta_fit,self.MB_fit = res['x'][0],res['x'][1],res['x'][2]
        self.res = res

    def get_mu_res_step(self,res,params):
        self.sim_df['mu_res'] = get_mu_res_step(res,self.sim_df,params,self.cosmo)
        self.sim_df['mu_res_err'] = self.sim_df['mB_err']
