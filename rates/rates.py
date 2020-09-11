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
import progressbar
import pystan
from des_sn_hosts.utils import stan_utility
import corner
import itertools

from des_sn_hosts.rates.rates_utils import sample_sn_masses, sample_field_masses
sns.set_color_codes(palette='colorblind')

class Rates():
    def __init__(self,config,fields=None,SN_hosts_fn=None, field_fn=None,origin='hostlib'):
        self.config =config
        self.SN_fn = SN_hosts_fn
        self.field_fn = field_fn
        if self.SN_fn:
            self.SN_Hosts = self._get_SN_hosts(SN_hosts_fn,fields)
        if self.field_fn:
            self.field = self._get_field(field_fn,origin)
        self.root_dir = self.config['rates_root']

    def _get_SN_hosts(self,fn,fields,):
        if fn.split('.')[-1]=='FITRES':

            df= pd.read_csv(fn,delimiter='\s+',comment='#').drop('VARNAMES:',axis=1)
            if fields:
                df =df[df['FIELD'].isin(fields)]
        elif fn.split('.')[-1]=='csv':
            df = pd.read_csv(fn,index_col=0)

        return df
    def _get_field(self,fn,origin):
        if origin == 'eazy':
            if fn.split('.')[-1]=='csv':
                return pd.read_csv(fn)
            elif fn.split('.')[-1]=='fits':
                zphot_res = Table.read(fn)
                zphot_res.remove_columns(['Avp','massp','SFRp','sSFRp','LIRp'])
                zphot_res = zphot_res.to_pandas()
            return zphot_res
        elif origin == 'hostlib':
            if fn.split('.')[-1]=='h5':
                return pd.read_hdf(fn,key='main')
            elif fn.split('.')[-1]=='fits':
                return Table.read(fn)
            elif fn.split('.')[-1]=='csv':
                return pd.read_csv(fn)
    def generate_sn_samples(self,mass_col='HOST_LOGMASS',mass_err_col='HOST_LOGMASS_ERR',
                                sfr_col = 'logssfr',sfr_err_col = 'logssfr_err',
                                index_col = 'CIDint',weight_col='weight',n_iter=1E5,save_samples=True):
        '''Wrapped around sample_sn_masses with option to save the output'''
        if weight_col not in self.SN_Hosts.columns:
            self.SN_Hosts[weight_col] = 1
        sn_samples = sample_sn_masses(self.SN_Hosts,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_col=mass_err_col,sfr_col=sfr_col,sfr_err_col=sfr_err_col,weight_col=weight_col,index_col=index_col,n_iter=n_iter)
        print('Sampling done')
        if save_samples:
            print('Saving to file')
            ext = '.'+self.SN_fn.split('.')[-1]
            savename=self.config['rates_root']+'data/'+os.path.split(self.SN_fn)[-1].replace(ext,'_mass_resampled.h5')
            sn_samples.to_hdf(savename,key='Bootstrap_samples')
        self.sn_samples = sn_samples

    def generate_field_samples(self,mass_col='mass',mass_err_col='mass_err',sfr_col = 'ssfr',sfr_err_col='ssfr_err',weight_col='weight',index_col = 'id',n_iter=1E5,save_samples=True):
        '''Wrapped around sample_sn_masses with option to save the output'''

        field_samples = sample_field_masses(self.field,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_col=mass_err_col,sfr_col = 'ssfr',sfr_err_col='ssfr_err',weight_col=weight_col,index_col=index_col,n_iter=n_iter)
        print('Sampling done')
        if save_samples:
            print('Saving to file')
            ext = '.'+self.field_fn.split('.')[-1]
            savename=self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_mass_resampled.h5')
            field_samples.to_hdf(savename,key='Bootstrap_samples')
        self.field_samples = field_samples

    def load_sn_samples(self):


        ext = '.'+self.SN_fn.split('.')[-1]
        savename=self.config['rates_root']+'data/'+os.path.split(self.SN_fn)[-1].replace(ext,'_mass_resampled.h5')

        self.sn_samples = sn_samples = pd.read_hdf(savename,key='Bootstrap_samples')

    def load_field_samples(self,mass_col='mass',err_col='mass_err',index_col = 'id',n_iter=1E5,save_samples=True):

        ext = '.'+self.field_fn.split('.')[-1]
        savename=self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_mass_resampled.h5')
        self.field_samples = pd.read_hdf(savename,key='Bootstrap_samples')

    def cut_z(self,z_min=0,z_max=1):
        self.sn_samples = self.sn_samples[(self.sn_samples['zHD']<z_max)&(self.sn_samples['zHD']>z_min)]
        self.field_samples = self.field_samples[(self.field_samples['z_phot']<z_max)&(self.field_samples['z_phot']>z_min)]

    def get_SN_bins(self,zmin=0,zmax=1.2,zstep=0.2,mmin=7.25,mmax=13,mstep=0.25):
        self.snzgroups = self.SN_Hosts.groupby(pd.cut(self.SN_Hosts.zHD,
                                                bins=np.linspace(zmin,zmax,((zmax-zmin)/zstep)+1)))['zHD']
        self.snmassgroups =self.SN_Hosts.groupby(pd.cut(self.SN_Hosts.HOST_LOGMASS,
                                                bins=np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)))['HOST_LOGMASS'] #,'VVmax'

    def get_field_bins(self,zmin=0,zmax=1.2,zstep=0.2,mmin=7.25,mmax=13,mstep=0.25):
        self.fieldzgroups =self.field.groupby(pd.cut(self.field.z_phot,
                                                bins=np.linspace(zmin,zmax,((zmax-zmin)/zstep)+1)))['z_phot']
        self.fieldmassgroups = self.field.groupby( pd.cut(self.field.mass,
                                                bins=np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)))['mass']

    def SN_G(self, scale='log',):
        '''Plots the SN/G rate for the data'''
        fmbinlog,axmbinlog = plt.subplots(figsize=(12,7))
        xs = []
        xerr = []
        ys = []
        yerr = []
        for (n,g),(n2,g2) in zip(self.snmassgroups,self.fieldmassgroups):
            if g.size >0 and g2.size>0:
                xs.append(n.mid)
                xerr.append(np.mean([np.abs(n.mid-n.left),np.abs(n.right-n.mid)]))
                ys.append(np.log10(g.size/g2.size)-1.38)
                YerrY = (((g.size)**(-1/4) + (g2.size)**(-1/4))**0.5)
                yerr.append(0.434*YerrY)
        xs = np.array(xs)
        ys = np.array(ys)
        xerr=np.array(xerr)
        yerr=np.array(yerr)
        counter=0
        for (n,g),(n2,g2) in zip(self.snmassgroups,self.fieldmassgroups):
            if g.size >0 and g2.size>0:

                axmbinlog.errorbar(n.mid,np.log10(g.size/g2.size)-1.38,
                           xerr=xerr[counter],
                           yerr=yerr[counter],
                            color='g',marker='D',label='All',
                                mew=0.3,mec='w',markersize=12)
                counter+=1
        axmbinlog.xaxis.set_minor_locator(MultipleLocator(0.25))
        axmbinlog.yaxis.set_minor_locator(MultipleLocator(0.125))
        axmbinlog.tick_params(which='both',right=True,top=True,direction='in',labelsize=16)
        axmbinlog.set_xlabel('Stellar Mass $\log (M_*/M_{\odot})$',size=20)
        axmbinlog.set_ylabel('$\log (N$ (SN hosts) / $N$ (Field Galaxies) )',size=20)

    def SN_G_MC(self,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,savename=None):
        mbins = np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)

        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =self.sn_samples.groupby(pd.cut(self.sn_samples[i],
                                                     bins=mbins))[i]
                i_f = np.random.randint(0,100)
                fieldmassgroups = self.field_samples.groupby( pd.cut(self.field_samples[i_f],
                                                            bins=mbins))[i_f]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g.weight.sum()/g2.weight.sum())-0.38) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples')
        self.sampled_rates = iter_df

    def SN_G_MC_SFR(self,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,sfr_cut_1=-11,sfr_cut_2=-9.5,savename=None):
        mbins = np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        # passive
        sn_passive = self.sn_samples[self.sn_samples['logssfr']<-11]
        field_passive = self.field_samples[self.field_samples['ssfr']<-11]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_passive.groupby(pd.cut(sn_passive[i],
                                                     bins=mbins))[i]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_passive.groupby( pd.cut(field_passive[i_f],
                                                            bins=mbins))[i_f]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g.size/g2.size)-0.38) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates_passive.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples')
        self.sampled_passive_rates = iter_df

        #moderately starforming
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        sn_moderate = self.sn_samples[(self.sn_samples['logssfr']>=-11)&(self.sn_samples['logssfr']<-9.5)]
        field_moderate = self.field_samples[(self.field_samples['ssfr']>=-11)&(self.field_samples['ssfr']<-9.5)]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_moderate.groupby(pd.cut(sn_moderate[i],
                                                     bins=mbins))[i]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_moderate.groupby( pd.cut(field_moderate[i_f],
                                                            bins=mbins))[i_f]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g.size/g2.size)-0.38) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates_moderate.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples')
        self.sampled_moderate_rates = iter_df

        #highly starforming
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        sn_high = self.sn_samples[self.sn_samples['logssfr']>=-9.5]
        field_high = self.field_samples[self.field_samples['ssfr']>=-9.5]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_high.groupby(pd.cut(sn_high[i],
                                                     bins=mbins))[i]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_high.groupby( pd.cut(field_high[i_f],
                                                            bins=mbins))[i_f]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g.size/g2.size)-0.38) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates_high.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples')
        self.sampled_high_rates = iter_df

    def load_sampled_rates(self,fn):
        self.sampled_rates = pd.read_hdf(fn,key='bootstrap_samples')

    def fit_SN_G(self,seed=123456,n_iter=4E3):

        model = stan_utility.compile_model(self.root_dir+'models/fit_yline_hetero.stan')
        x_model = np.linspace(6.5,11,100)
        x_obs = np.array(self.sampled_rates.index)[2:-6]
        y_obs = self.sampled_rates.mean(axis=1).values[2:-6]
        y_err = self.sampled_rates.std(axis=1).values[2:-6]

        data = dict(N = len(x_obs),
                    x_obs = x_obs,
                    y_obs = y_obs,
                    #sigma_x=np.array(xerr[:-2]),
                    sigma=y_err,
                    N_model=100,
                   x_model=x_model)
        fit = model.sampling(data=data, seed=seed, iter=n_iter)
        return fit

    def plot_fit(self,fit):
        fmbinlog,axmbinlog = plt.subplots(figsize=(12,7))
        chain = fit.extract()
    # Plot the points from above as a comparison
        x_model = np.linspace(6.5,11,100)
        for counter,c in enumerate(self.sampled_rates.columns):
            label=None
            if counter == 0:
                label='Observations'
            axmbinlog.scatter(self.sampled_rates.index,self.sampled_rates[c],color='g',marker='o',
                           alpha=0.05,s=10,label=label)
            axmbinlog.xaxis.set_minor_locator(MultipleLocator(0.25))
            axmbinlog.yaxis.set_minor_locator(MultipleLocator(0.125))
            axmbinlog.tick_params(which='both',right=True,top=True,direction='in',labelsize=16)
            axmbinlog.set_xlabel('Stellar Mass $\log (M_*/M_{\odot})$',size=20)
            axmbinlog.set_ylabel('$\log (N$ (SN hosts) / $N$ (Field Galaxies) )',size=20)
        for i in self.sampled_rates.index:
            axmbinlog.errorbar(i,self.sampled_rates.loc[i].mean(),xerr=0.184,color='g',marker='D',
                           alpha=0.5,markersize=2,mew=0.5,mec='w')

        level = 95

        axmbinlog.fill_between(x_model,
                        np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                        np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                        color='c',alpha=0.2)

        level = 68
        axmbinlog.fill_between(x_model,
                        np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                        np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                        color='c',alpha=0.3)

        axmbinlog.plot(x_model,
                        np.percentile(chain['line'], 50, axis=0 ),
                        color='b',alpha=1,linestyle='-',linewidth=1,label='$dN/dG = %.2f$'%np.median(chain['slope']))
        leg =axmbinlog.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.savefig(self.root_dir +'figs/rate_vs_mass_slopes_stanfit.png')
