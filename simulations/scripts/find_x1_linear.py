# Imports
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
import pickle
from scipy.interpolate import interp1d
#import eazy
#eazy.symlink_eazy_inputs()
#eazy_dir = os.getenv('EAZYCODE')
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

sns.set_color_codes(palette='colorblind')
import time
import _pickle as cpickle
import itertools
import progressbar
from tqdm import tqdm
from yaml import safe_load as yload
plt.rcParams['errorbar.capsize']=4

plt.style.use('default')
#plt.style.use('dark_background')
sns.set_context('paper')
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': [16,9]})
plt.rcParams.update({'xtick.direction':'in'})
plt.rcParams.update({'ytick.direction':'in'})
plt.rcParams['text.usetex'] = True
#plt.rcParams.update({'figure.facecolor':'#16191C'})
#plt.rcParams.update({'axes.facecolor':'#16191C'})
split_colour_1 = '#f28500'
split_colour_2 = '#8500f2'

from astropy.cosmology import FlatLambdaCDM
cosmo =FlatLambdaCDM(H0=70,Om0=0.3)

aura_dir = '/media/data3/wiseman/des/AURA/'
from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter_paper import *
from des_sn_hosts.simulations.utils.HR_functions import calculate_step, get_red_chisq

from des_sn_hosts.simulations.utils.plotter_paper import *
from des_sn_hosts.simulations.utils.plotter import *
#des5yr = pd.read_csv('/media/data3/wiseman/des/AURA/data/df_after_cuts_z0.6_UR1.csv')
des5yr = pd.read_hdf('/media/data3/wiseman/des/AURA/data/DES5YR_MV20200701_Hosts20211018.h5')
def get_chi2_x1(df,bin_centers,des5yr,means,stds,norm_len='none'):
    
    simcounts_chi2,simbins_chi2 = np.histogram(df['x1'],density=False,bins=np.linspace(-5,5,30))
    sim_bins_chi2 = (simbins_chi2[:-1] + simbins_chi2[1:])/2
    simcounts_chi2 = simcounts_chi2 * len(des5yr)/norm_len * (bin_centers[-1]-bin_centers[-2])/(simbins_chi2[-1]-simbins_chi2[-2])
    chi2x1 = get_red_chisq(means,simcounts_chi2,stds)
    return chi2x1

def run_chi2(df,des5yr):
    means,bins = np.histogram(des5yr['x1'],bins=np.linspace(-5,5,30))
    stds =np.clip(np.sqrt(means),a_min=1,a_max=None)
    bin_centres_x1 = (bins[1:]+bins[:-1])/2
    norm_len = len(df)
    chi2  = get_chi2_x1(df,bin_centres_x1,des5yr,means,stds,norm_len,)
    return chi2
sim_linear_x1= aura.Sim('/home/wiseman/code/des_sn_hosts/simulations/config/DES_BS20_age_Rv_step_3Gyr_age_x1_beta_1.14_x1_linear.yaml')

slope_bins = np.arange(-1.5,-0.5,0.1)
width_bins = np.arange(0.1,1,0.2)
offset_bins = np.arange(-1,0,0.2)
chi2s = np.zeros((len(slope_bins),len(width_bins),len(offset_bins)))
for i,s in enumerate(slope_bins):
    for j,w in enumerate(width_bins):
        for k,o in enumerate(offset_bins):
            sim_linear_x1.config['x1_model']['params']['slope']=s
            sim_linear_x1.config['x1_model']['params']['width']=w
            sim_linear_x1.config['x1_model']['params']['offset']=o
            n_samples=5000
            zs = np.linspace(0,1,100)
            zs_cubed = zs**2
            numbers = np.random.choice(zs,p=zs_cubed/np.sum(zs_cubed),size=n_samples)
            n_samples_arr = sim_linear_x1._get_z_dist(numbers,n=n_samples,frac_low_z=0)
            zarr=[0.05,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,] #0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2
            #AV 0!!
            #sim.flux_df['Av']=float(0)

            sim_linear_x1.sample_SNe(zarr,n_samples_arr,savepath='/media/data3/wiseman/des/AURA/sims/SNe/DES_BS20_age_Rv_step_3Gyr_age_x1_beta_1.14_x1_linear_int_%.2f_%.2f_%.2f_SN_sim.h5'%(s,w,o))
            fn = '/media/data3/wiseman/des/AURA/sims/SNe/DES_BS20_age_Rv_step_3Gyr_age_x1_beta_1.14_x1_linear_int_%.2f_%.2f_%.2f_SN_sim.h5'%(s,w,o)
            sim_linear_x1.load_sim(fn)
            sim = sim_linear_x1
            des5yr = pd.read_hdf(os.path.join(aura_dir,'data','DES5YR_MV20200701_Hosts20211018_BBC1D.h5'))
            sim_linear_x1.load_sim(fn)
            sim = sim_linear_x1
            sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&\
                                        (sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&\
                                        (sim.sim_df['c_err']<0.1)   # uncomment to include a colour error cut
                                        ]
            sim.sim_df = sim.sim_df[sim.sim_df['mB']<25]
            sim.sim_df = sim.sim_df[sim.sim_df['eff_mask']==1]
            sim.sim_df = sim.sim_df[sim.sim_df['z']>=0.15]
            sim.sim_df = sim.sim_df[sim.sim_df['z']<0.71]
            chi2 = run_chi2(sim.sim_df,des5yr)
            chi2s[i,j,k] = chi2
            print("adding chi2 for",i,j,k)
np.save('/media/data3/wiseman/des/AURA/sims/SNe/x1s/linear_chi2s.npy',chi2s)
