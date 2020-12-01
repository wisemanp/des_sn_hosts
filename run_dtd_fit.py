import numpy as np
import pandas as pd
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from astropy.coordinates import SkyCoord
import logging
from astropy.table import Table
import astropy.io.fits as fits
import os
import scipy.stats as stats
from shutil import copyfile
from astropy import units as u
from astropy import wcs
import aplpy
import numpy as np
import subprocess
import glob
import matplotlib.pyplot as plt
import os
import astropy
import sys

import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

sns.set_color_codes(palette='colorblind')
import time
import _pickle as cpickle
import itertools
import progressbar
from tqdm import tqdm

plt.rcParams['errorbar.capsize']=4
#plt.style.use('dark_background')
plt.style.use('default')
sns.set_context('paper')
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': [16,9]})
plt.rcParams.update({'xtick.direction':'in'})
plt.rcParams.update({'ytick.direction':'in'})

from des_sn_hosts.simulations.sim import ZPowerCosmoSchechterSim
from des_sn_hosts.utils.utils import Constants
c = Constants()
from astropy.cosmology import WMAP9 as cosmo
import confuse

from des_sn_hosts.rates.rates import Rates
import yaml
import numpy as np

sn_hosts_fn ='/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701_FITRES_merged_SNR3_weighted_vmax_cuts_sedfit_BC03_noneb_mc.h5'  #'/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701_FITRES_merged_SNR3_weighted_cuts_MS_ZPEG.h5'
X3_hostlib_fn = '/media/data3/wiseman/des/coadding/results/deep/deep_X3_hostlib_sedfit_BC03_vmax_mc.h5'
r_BC03_noneb = Rates(SN_hosts_fn=sn_hosts_fn,field_fn=X3_hostlib_fn,
          config=yaml.load(open('/home/wiseman/code/des_sn_hosts/config/config_rates.yaml')))
r_BC03_noneb.field.rename(columns={'redshift_sedfit':'redshift'},inplace=True)
r_BC03_noneb.rate_corr-= np.log10(59/46)
r_BC03_noneb.sampled_rates_mass_fine_BC03 = pd.read_hdf(r_BC03_noneb.config['rates_root']+'data/mcd_rates_BC03_noneb.h5',key='bootstrap_samples_mass_0.25')
store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_0.5.h5','r')

ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])
model_df = pd.DataFrame(index = store['/'+str(ordered_keys[-1])]['age'].values[::-1])
z =0.55
for counter,tf in enumerate(ordered_keys[::-1]):   # Iterate through the SFHs for galaxies of different final masses
    #print(tf)
    sfh_df = store['/'+str(tf)]
    sfh_df = sfh_df[sfh_df['z']>z]
    if len(sfh_df)>0:
        m = sfh_df['m_tot'].iloc[-1]
        ts =sfh_df['age'].values[::-1]
        model_df[m] = 0

        model_df[m].loc[ts] = sfh_df['m_formed'].values

obs = r_BC03_noneb.sampled_rates_mass_fine_BC03.dropna(axis=0)
model_ms = np.array(model_df.columns.tolist())
fitting_arr = np.zeros((len(obs.index),len(model_df.index)))
#f,ax=plt.subplots()
for counter,m in enumerate(10**obs.index):

    argmin = np.argmin(np.abs(m-model_ms))
    m1 =model_ms[argmin]
    min_diff = model_ms[argmin] - m

    if min_diff >0:
        m2 = model_ms[argmin+1]

    else:
        m2 = model_ms[argmin-1]
    frac_diff = (m-m1 )/(m2-m1)

    ms_interp = model_df[m1] + frac_diff*(model_df[m2] - model_df[m1])
    fitting_arr[counter,:] = ms_interp
from des_sn_hosts.utils import stan_utility

model = stan_utility.compile_model('/home/wiseman/code/des_sn_hosts/'+'models/fit_dtd.stan')
x_model = np.linspace(7,12,100)

data = dict(N = len(fitting_arr),
            M = len(model_df.index),
            age = model_df.index.values/1000,
            SFH = fitting_arr,
            logmass_obs = obs.index,
            lograte_obs = obs[np.arange(0,100)].median(axis=1).values,
            sigma = obs[np.arange(0,100)].std(axis=1),
            )
fit = model.sampling(data=data, seed=1234, iter=int(1000),
        warmup=500,sample_file = r_BC03_noneb.config['rates_root']+'/data/dtd_samples.dat')
print(fit)
