import numpy as np
import pandas as pd
import os
import astropy
import argparse
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

sns.set_color_codes(palette='colorblind')
from des_sn_hosts.rates.rates import Rates


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cut',help='Which lc cut to do',default='None',type=float)
    args = parser.parse_args()
    return args
args = parser()

sn_hosts_fn ='/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701_FITRES_merged_SNR3_eff_vmax_cuts_sedfit_BC03_noneb_mc.h5'  #'/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701_FITRES_merged_SNR3_weighted_cuts_MS_ZPEG.h5'
X3_hostlib_fn = '/media/data3/wiseman/des/coadding/results/deep/deep_X3_hostlib_finalz_sedfit_BC03_vmax_mc.h5'
r_BC03_noneb = Rates(SN_hosts_fn=sn_hosts_fn,field_fn=X3_hostlib_fn,
          config=yaml.load(open('/home/wiseman/code/des_sn_hosts/config/config_rates.yaml')))
r_BC03_noneb.field.rename(columns={'redshift_sedfit':'redshift'},inplace=True)
r_BC03_noneb.rate_corr= np.log10(1.52/27)
des_end = 58178
des_start = 56535
des_time = (des_end - des_start)/365.25
r_BC03_noneb.rate_corr -= np.log10(des_time)

r_BC03_noneb.sampled_rates_mass_fine_BC03_lccut = pd.read_hdf(r_BC03_noneb.config['rates_root']+'data/mcd_rates_BC03_noneb_eff_3d_finalz_V20cuts_zmax06_%s.h5'%args.cut,key='bootstrap_samples_mass_0.25')

store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_alt_0.5_Qerf_1.1.h5','r')

ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])
model_df = pd.DataFrame(index = store['/'+str(ordered_keys[-1])]['age'].values[::-1])
z =0.5
for counter,tf in enumerate(ordered_keys[::-1]):   # Iterate through the SFHs for galaxies of different final masses
    sfh_df = store['/'+str(tf)]
    sfh_df = sfh_df[sfh_df['z']>z]
    if len(sfh_df)>0:
        m = sfh_df['m_tot'].iloc[-1]
        ts =sfh_df['age'].values[::-1]
        model_df[m] = 0
        model_df[m].loc[ts] = sfh_df['m_formed'].values
obs = r_BC03_noneb.sampled_rates_mass_fine_BC03_lccut.replace([np.inf,-np.inf],np.NaN).dropna(axis=0,how='any')#.loc[9.:11.5]
model_ms = np.array(model_df.columns.tolist())
fitting_arr = np.zeros((len(model_df.index),len(obs_x1hi.index)))
for counter,m in enumerate(10**obs_x1hi.index):
    argmin = np.argmin(np.abs(m-model_ms))
    m1 =model_ms[argmin]
    min_diff = model_ms[argmin] - m
    if min_diff >0:
        m2 = model_ms[argmin+1]
    else:
        m2 = model_ms[argmin-1]
    frac_diff = (m-m1 )/(m2-m1)
    ms_interp = model_df[m1] + frac_diff*(model_df[m2] - model_df[m1])
    fitting_arr[:,counter] = ms_interp
from des_sn_hosts.utils import stan_utility

model = stan_utility.compile_model('/home/wiseman/code/des_sn_hosts/'+'models/fit_dtd_simple_pl_find_tp.stan')
x_model = np.linspace(7,12,100)

data = dict(N = len(fitting_arr),
            M = len(model_df.index),
            age = model_df.index.values/1000,
            SFH = fitting_arr,
            logmass_obs = obs.index,
            lograte_obs = obs[np.arange(0,100)].median(axis=1).values,
            sigma = obs[np.arange(0,100)].std(axis=1),
            )
fit = model.sampling(data=data, seed=1234, iter=int(2000),
    warmup=1000,sample_file = r_BC03_noneb.config['rates_root']+'/data/dtd_samples_with_eff_finalz_tp_%s'%args.cut)
df = fit.to_dataframe()
df.to_hdf(r_BC03_noneb.config['rates_root']+'/data/dtd_samples_with_eff_finalz_tp_%s.h5'%args.cut,key='samples')
print(fit)
