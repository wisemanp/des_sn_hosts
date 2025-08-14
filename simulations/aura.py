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
from .utils.gal_functions import schechter, single_schechter, double_schechter, ozdes_efficiency, interpolate_zdf, make_z_pdf
#from .utils.plotter import *
from .utils.HR_functions import get_mu_res_step, get_mu_res_nostep, chisq_mu_res_nostep, chisq_mu_res_step,chisq_mu_res_nostep_old

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import itertools
from tqdm import tqdm


aura_dir = os.environ['AURA_DIR']
idx = pd.IndexSlice

age_grid = np.arange(0,13.7,0.0005)
age_grid_index = ['%.4f'%a for a in age_grid]
class Sim(SN_Model):
    """
    Simulation class for drawing SN samples from a galaxy population.
    """

    def __init__(self, conf_path, cosmo='default'):
        self.config = self._get_config(conf_path)

        root_dir = self.config['config']['root_dir']
        self.root_dir = os.environ.get(root_dir[1:]) if root_dir.startswith('$') else root_dir

        self.fig_dir = os.path.join(self.root_dir, 'figs/')
        self.eff_dir = self.config['config']['efficiency_dir']
        self.flux_df = self._load_flux_df(self.config['hostlib_fn'])

        self._calculate_absolute_rates()
        self._make_multi_index()

        self.cosmo = FlatLambdaCDM(70, 0.3) if cosmo == 'default' else cosmo
        self._get_funcs()

    # ----------------------
    # CONFIG & DATA LOADING
    # ----------------------
    def _get_config(self, conf_path):
        with open(conf_path, 'r') as f:
            return yload(f)

    def _load_flux_df(self, fn):
        df = pd.read_hdf(fn)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                print(f"Non-numeric column in flux_df: {col}")
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
            return pd.DataFrame()

        rng = np.random.default_rng()
        args = {'n': int(n_samples), 'distmod': self.cosmo.distmod(z).value}

        # Get galaxies at this redshift
        z_df = self.multi_df.loc[f"{z:.5f}"].copy()
        z_df.replace({'N_total': {0.0: np.nan}}, inplace=True)
        z_df.dropna(subset=['N_total'], inplace=True)
        z_df['N_SN_float'] = z_df['N_total'] / z_df['N_total'].min()
        z_df['N_SN_int'] = z_df['N_SN_float'].astype(int)

        # Interpolate over masses for each Av
        marr = np.logspace(6, 11.6, 100)
        resampled_df = pd.DataFrame()
        for av in z_df.Av.unique():
            av_df = z_df.loc[idx[:, f"{av:.5f}", :]]
            print(av_df)
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
            if len(g) == 0:
                continue
            min_av = g.Av.astype(float).min()
            g_Av_0 = g.loc[idx[:, f"{min_av:.5f}", :]]

            age_df = pd.DataFrame(index=age_grid_index)
            for k in g_Av_0.index.unique():
                sub_gb = g_Av_0.loc[k]
                tf = sub_gb['t_f'].iloc[0] if isinstance(sub_gb, pd.DataFrame) else sub_gb['t_f']
                split_z = os.path.split(self.config['hostlib_fn'])[1].split('z')
                split_rv = os.path.split(self.config['hostlib_fn'])[1].split('rv')
                ext = f"{split_z[0]}z_{z:.5f}_rv{split_rv[1][:-12]}_{tf:.1f}_combined.dat"
                new_fn = os.path.join(os.path.split(self.config['hostlib_fn'])[0], 'SN_ages', ext)
                sub_gb = pd.read_csv(new_fn, sep=' ', names=['SN_ages', 'SN_age_dist'])
                age_inds = [f"{a:.4f}" for a in sub_gb['SN_ages']]
                age_df.loc[age_inds, f"{float(k):.2f}"] = (
                    sub_gb['SN_age_dist'].values / np.nansum(sub_gb['SN_age_dist'].values)
                )

            age_df.fillna(0, inplace=True)
            avg_dist = np.nanmean(age_df, axis=1).values

            # Assign to all matching rows
            for idx_key in new_zdf.index[new_zdf['mass'].between(
                g.mass.min(), g.mass.max(), inclusive='both')]:
                new_zdf.at[idx_key, 'SN_age_dist'] = avg_dist.copy()

        # Select galaxies & sample ages
        m_inds = new_zdf.index.get_level_values(0).unique()
        m_rates = new_zdf.groupby(level=0)['N_SN_int'].first().values
        m_samples = rng.choice(m_inds, p=m_rates / np.sum(m_rates), size=int(n_samples))
        m_av0_samples = [(m, f"{rng.choice(new_zdf.loc[m].Av.values):.5f}") for m in m_samples]

        sn_ages = []
        for m_av in m_av0_samples:
            probs = new_zdf.loc[m_av, 'SN_age_dist']
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
            sn_ages.append(rng.choice(new_zdf.loc[m_av, 'SN_ages'], p=probs))

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
        args['c_noise'] = norm(0, args['c_err']).rvs(size=len(args['c']))
        if not self.config['c_perfect']:
            args['c'] = args['c'] + args['c_noise']

        args['x1_err'] = [np.max([0.08, np.random.normal((11.525*err - 0.1075), 0.05)])
                          for err in args['mB_err']]
        args['x1_noise'] = norm(0, args['x1_err']).rvs(size=len(args['x1']))
        args['x1_int'] = args['x1'].copy()
        args['x1'] = args['x1'] + args['x1_noise']
        args['cov_mB_x1'], args['cov_mB_c'], args['cov_x1_c'] = 0, 0, 0

        args['distmod'] = np.ones_like(args['c']) * args['distmod']
        del args['Av_grid']

        z_sim_df = pd.DataFrame(args)
        z_sim_df['z'] = z
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
