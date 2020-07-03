import numpy as np
import pandas as pd
import subprocess
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.io.fits as fits
import os
import sys
from shutil import copyfile
from scipy import stats
from scipy.optimize import curve_fit
import pystan
from utils import stan_utility
import corner
import itertools

from des_sn_hosts.rates.rates_utils import sample_masses
sns.set_color_codes(palette='colorblind')

class Rates():
    def __init__(self,SN_hosts_fn, field_fn,config):
        self.config =config
        self.SN_fn = SN_hosts_fn
        self.field_fn = field_fn
        self.SN_Hosts = self._get_SN_hosts(SN_hosts_fn)
        self.field = self._get_SN_hosts(field_fn)


    def _get_SN_hosts(self,fn):
        return pd.read_csv(fn)

    def _get_field(self,fn):
        return pd.read_csv(fn)

    def generate_sn_samples(self,mass_col='log_m',err_col='logm_err',index_col = 'CIDint',n_iter=1E4,save_samples=True):
        '''Wrapped around sample_sn_masses with option to save the output'''

        sn_samples = sample_sn_masses(self.SN_hosts,mass_col=mass_col,err_col=err_col,index_col=index_col,n_iter=n_iter)
        if save_samples:
            savename=self.config['rates_root']+self.SN_fn.replace('.csv','_mass_resampled.h5'))
            sn_samples.to_hdf(savename,key='Bootstrap_samples')
        self.sn_samples = sn_samples

    def generate_sn_samples(self,mass_col='mass',err_col='mass_err',index_col = 'id',n_iter=1E4,save_samples=True):
        '''Wrapped around sample_sn_masses with option to save the output'''

        sn_samples = sample_sn_masses(self.field,mass_col=mass_col,err_col=err_col,index_col=index_col,n_iter=n_iter)
        if save_samples:
            savename=self.config['rates_root']+self.SN_fn.replace('.csv','_mass_resampled.h5'))
            sn_samples.to_hdf(savename,key='Bootstrap_samples')
        self.sn_samples = sn_samples
