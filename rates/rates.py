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

from des_sn_hosts.rates.rates_utils import sample_sn_masses, sample_field_masses, sample_field_asymm, split_by_z, SN_G_MC
sns.set_color_codes(palette='colorblind')

class Rates():
    def __init__(self,config,fields=None,SN_hosts_fn=None, field_fn=None,origin='hostlib',N_SN_fields=10,N_field_fields=1):
        self.config =config
        self.SN_fn = SN_hosts_fn
        self.field_fn = field_fn
        if self.SN_fn:
            self.SN_Hosts = self._get_SN_hosts(SN_hosts_fn,fields)
        if self.field_fn:
            self.field = self._get_field(field_fn,origin)
        self.root_dir = self.config['rates_root']

        self._get_rate_corr(N_SN_fields,N_field_fields)
    def _get_SN_hosts(self,fn,fields,):
        if fn.split('.')[-1]=='FITRES':

            df= pd.read_csv(fn,delimiter='\s+',comment='#').drop('VARNAMES:',axis=1)
            if fields:
                df =df[df['FIELD'].isin(fields)]
        elif fn.split('.')[-1]=='csv':
            df = pd.read_csv(fn,index_col=0)

        elif fn.split('.')[-1]=='h5':
            df = pd.read_hdf(fn,key='main')

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

    def _get_rate_corr(self,N_SN_fields,N_field_fields):
        self.rate_corr = -0.38 -np.log10(N_SN_fields/N_field_fields)
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
        self.sn_samples_mass = sn_samples

    def generate_field_samples(self,mass_col='mass',mass_err_col='mass_err',mass_err_plus = 'MASSMAX',mass_err_minus='MASSMIN',
    sfr_col = 'log_sfr',sfr_err_col='log_sfr_err',sfr_err_plus='SFRMAX',sfr_err_minus='SFRMIN',
    weight_col='weight',index_col = 'id',n_iter=1E5,save_samples=True,asymm=False):
        '''Wrapped around sample_sn_masses with option to save the output'''

        if not asymm:
            field_samples = sample_field_masses(self.field,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_col=mass_err_col,sfr_col = sfr_col,sfr_err_col=sfr_err_col,weight_col=weight_col,index_col=index_col,n_iter=n_iter,variable='mass')
        else:
            field_samples = sample_field_masses_asymm(self.field,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_plus=mass_err_plus,mass_err_minus=mass_err_minus,
                    sfr_col = sfr_col,sfr_err_plus=sfr_err_plus,sfr_err_minus = sfr_err_minus,
                    weight_col=weight_col,index_col=index_col,n_iter=n_iter,variable='mass')
        print('Sampling done')
        if save_samples:
            print('Saving to file')
            ext = '.'+self.field_fn.split('.')[-1]
            savename=self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_mass_resampled.h5')
            field_samples.to_hdf(savename,key='Bootstrap_samples')
        self.field_samples_mass = field_samples

    def generate_sn_samples_sfr(self,mass_col='HOST_LOGMASS',mass_err_col='HOST_LOGMASS_ERR',
                                sfr_col = 'logsfr',sfr_err_col = 'logsfr_err',
                                index_col = 'CIDint',weight_col='weight',n_iter=1E5,save_samples=True):
        '''Wrapped around sample_sn_masses with option to save the output'''
        if weight_col not in self.SN_Hosts.columns:
            self.SN_Hosts[weight_col] = 1
        sn_samples = sample_sn_masses(self.SN_Hosts,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_col=mass_err_col,sfr_col=sfr_col,sfr_err_col=sfr_err_col,weight_col=weight_col,index_col=index_col,n_iter=n_iter,variable='sfr')
        print('Sampling done')
        if save_samples:
            print('Saving to file')
            ext = '.'+self.SN_fn.split('.')[-1]
            savename=self.config['rates_root']+'data/'+os.path.split(self.SN_fn)[-1].replace(ext,'_sfr_resampled.h5')
            sn_samples.to_hdf(savename,key='Bootstrap_samples')
        self.sn_samples_sfr = sn_samples

    def generate_field_samples_sfr(
        self,mass_col='mass',mass_err_col='mass_err',mass_err_plus = 'MASSMAX',mass_err_minus='MASSMIN',
        sfr_col = 'log_sfr',sfr_err_col='log_sfr_err',sfr_err_plus='SFRMAX',sfr_err_minus='SFRMIN',
        weight_col='weight',index_col = 'id',n_iter=1E5,save_samples=True,asymm=False
    ):
        '''Wrapped around sample_sn_masses with option to save the output'''

        if not asymm:
            field_samples = sample_field_masses(self.field,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_col=mass_err_col,sfr_col = sfr_col,sfr_err_col=sfr_err_col,weight_col=weight_col,index_col=index_col,n_iter=n_iter,variable='sfr')
        else:
            field_samples = sample_field_masses_asymm(self.field,self.config['rates_root']+'models/',
                    mass_col=mass_col,mass_err_plus=mass_err_plus,mass_err_minus=mass_err_minus,
                    sfr_col = sfr_col,sfr_err_plus=sfr_err_plus,sfr_err_minus = sfr_err_minus,weight_col=weight_col,index_col=index_col,n_iter=n_iter,variable='sfr')

        print('Sampling done')
        if save_samples:
            print('Saving to file')
            ext = '.'+self.field_fn.split('.')[-1]
            savename=self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_sfr_resampled.h5')
            field_samples.to_hdf(savename,key='Bootstrap_samples')
        self.field_samples_sfr = field_samples

    def generate_samples_split_z(self,zmin,zmax,zstep,variable='mass'):
        sn_samples_z = {}
        field_samples_z = {}
        for zlo in np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False):
            zhi=zlo+zstep
            # SN Hosts
            sn_df = pd.read_hdf(self.SN_fn,key='z_%.2f_%.2f'%(zlo,zhi))
            sn_sample = sample_sn_masses(sn_df,self.config['rates_root']+'models/',
                                                mass_col='HOST_LOGMASS',mass_err_col='HOST_LOGMASS_ERR',
                                                        sfr_col = 'logsfr',sfr_err_col = 'logsfr_err',
                                                        index_col = 'CIDint',weight_col='weight',n_iter=int(1E4),variable=variable)
            ext = '.'+self.SN_fn.split('.')[-1]
            sn_sample.to_hdf(self.config['rates_root']+'data/'+os.path.split(self.SN_fn)[-1].replace(ext,'_%s_resampled.h5'%variable),key='Bootstrap_samples_z_%.2f_%.2f'%(zlo,zhi))
            sn_samples_z['%.2f-%.2f'%(zlo,zhi)] = sn_sample
            field_df = pd.read_hdf(self.field_fn,key='z_%.2f_%.2f'%(zlo,zhi))
            field_sample = sample_field_asymm(field_df,self.config['rates_root']+'models/',sfr_col='SFR',sfr_err_plus='SFRMAX',sfr_err_minus='SFRMIN',weight_col='VVmax',index_col = 'id',n_iter=int(1E4),variable=variable)
            ext = '.'+self.field_fn.split('.')[-1]
            field_sample.to_hdf(self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_%s_resampled.h5'%variable),key='Bootstrap_samples_z_%.2f_%.2f'%(zlo,zhi))
            field_samples_z['%.2f-%.2f'%(zlo,zhi)] = field_sample
        setattr(self,'sn_samples_%s_z'%variable,sn_samples_z)
        setattr(self,'field_samples_%s_z'%variable,field_samples_z)
        return sn_samples_z, field_samples_z

    def load_sn_samples(self,variable = 'mass',key_ext=None):
        ext = '.'+self.SN_fn.split('.')[-1]
        savename=self.config['rates_root']+'data/'+os.path.split(self.SN_fn)[-1].replace(ext,'_%s_resampled.h5'%variable)
        if not key_ext:
            setattr(self,'sn_samples_%s'%variable,pd.read_hdf(savename,key='Bootstrap_samples'))
        else:
            setattr(self,'sn_samples_%s_%s'%(variable,key_ext),pd.read_hdf(savename,key='Bootstrap_samples_%s'%key_ext))

    def load_field_samples(self,variable='mass',key_ext=None):

        ext = '.'+self.field_fn.split('.')[-1]
        savename=self.config['rates_root']+'data/'+os.path.split(self.field_fn)[-1].replace(ext,'_%s_resampled.h5'%variable)
        if not key_ext:
            setattr(self,'field_samples_%s'%variable,pd.read_hdf(savename,key='Bootstrap_samples'))
        else:
            setattr(self,'field_samples_%s_%s'%(variable,key_ext),pd.read_hdf(savename,key='Bootstrap_samples_%s'%key_ext))

    def cut_z(self,z_min=0,z_max=1):
        for sn_samples,field_samples in zip([self.sn_samples_mass,self.sn_samples_sfr],[self.field_samples_mass,self.field_samples_sfr]):

            sn_samples = sn_samples[(sn_samples['zHD']<z_max)&(sn_samples['zHD']>z_min)]
            field_samples = field_samples[(field_samples['redshift']<z_max)&(field_samples['redshift']>z_min)]

    def split_by_z(self,zmin=0.2,zmax=1.2,zstep=0.2):
        split_by_z(self.SN_Hosts,self.SN_fn,zmin=zmin,zmax=zmax,zstep=zstep)
        split_by_z(self.field,self.field_fn,zcol='redshift',zmin=zmin,zmax=zmax,zstep=zstep,do_VVmax=True)

    def get_SN_bins(self,zmin=0,zmax=1.2,zstep=0.2,mmin=7.25,mmax=13,mstep=0.25):
        self.snzgroups = self.SN_Hosts.groupby(pd.cut(self.SN_Hosts.zHD,
                                                bins=np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False)))['zHD']
        self.snmassgroups =self.SN_Hosts.groupby(pd.cut(self.SN_Hosts.HOST_LOGMASS,
                                                bins=np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False)))['HOST_LOGMASS'] #,'VVmax'

    def get_field_bins(self,zmin=0,zmax=1.2,zstep=0.2,mmin=7.25,mmax=13,mstep=0.25,mass_col='log_mass'):
        self.fieldzgroups =self.field.groupby(pd.cut(self.field.redshift,
                                                bins=np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False)))['redshift']
        self.fieldmassgroups = self.field.groupby( pd.cut(self.field[mass_col],
                                                bins=np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False)))[mass_col]

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
                ys.append(np.log10(g.size/g2.size)+self.rate_corr)
                YerrY = (((g.size)**(-1/4) + (g2.size)**(-1/4))**0.5)
                yerr.append(0.434*YerrY)
        xs = np.array(xs)
        ys = np.array(ys)
        xerr=np.array(xerr)
        yerr=np.array(yerr)
        counter=0
        for (n,g),(n2,g2) in zip(self.snmassgroups,self.fieldmassgroups):
            if g.size >0 and g2.size>0:

                axmbinlog.errorbar(n.mid,np.log10(g.size/g2.size)+self.rate_corr,
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

    def SN_G_MC(self,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,savename=None, weight_col_SN='weight',weight_col_field='weight'):

        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates.h5'
        iter_df = SN_G_MC(self.sn_samples_mass,self.field_samples_mass,n_samples=n_samples,mmin=mmin,mmax=mmax,mstep=mstep,savename=savename, weight_col_SN=weight_col_SN,weight_col_field=weight_col_field,key_ext='%.2f'%mstep)
        self.sampled_rates_mass = iter_df



    def SN_G_MC_z(self,zmin=0.2,zmax=0.8,zstep=0.2,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,savename=None, weight_col_SN='weight',weight_col_field='weight'):
        print('Calculating rates for redshift bins: ',np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False))
        for zlo in np.linspace(zmin,zmax,int((zmax-zmin)/zstep),endpoint=False):
            zhi = zlo+zstep
            print(zlo,'-',zhi)
            key = 'z_%.2f_%.2f_%.2f'%(zlo,zhi,mstep)
            sn_df = getattr(self,'sn_samples_mass_%s'%key)
            field_df = getattr(self,'field_samples_mass_%s'%key)
            savename=self.config['rates_root']+'data/mcd_rates.h5'
            iter_df = SN_G_MC(sn_df,field_df,n_samples=int(1E+2),mmin=mmin,mmax=mmax,mstep=mstep,savename=savename, variable='mass',key_ext=key,weight_col_SN='weight',weight_col_field='VVmax',rate_corr = self.rate_corr)
            setattr(self,'sampled_rates_mass_%s'%key,iter_df)
        return iter_df
    def SN_G_MC_SFR(self,n_samples=1E4,sfrmin=-3,sfrmax=2,sfrstep=0.25,savename=None, weight_col_SN='weight',weight_col_field='weight'):
        mbins = np.linspace(sfrmin,sfrmax,((sfrmax-sfrmin)/sfrstep)+1)
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)

        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snsfrgroups =self.sn_samples_sfr.groupby(pd.cut(self.sn_samples_sfr[i],
                                                     bins=mbins))[[i,weight_col_SN]]
                i_f = np.random.randint(0,100)
                fieldsfrgroups = self.field_samples_sfr.groupby( pd.cut(self.field_samples_sfr[i_f],
                                                            bins=mbins))[[i_f,weight_col_field]]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snsfrgroups,fieldsfrgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g[weight_col_SN].sum()/g2[weight_col_field].sum())+self.rate_corr) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_sfr')
        self.sampled_rates_sfr = iter_df

    def SN_G_MC_MASS_SFR(self,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,sfr_cut_1=-11,sfr_cut_2=-9.5, sn_ssfr_col = 'logssfr', field_ssfr_col='SPECSFR', savename=None,weight_col_SN='weight',weight_col_field='weight'):
        mbins = np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        # passive
        sn_passive = self.sn_samples[self.sn_samples[sn_ssfr_col]<sfr_cut_1]
        field_passive = self.field_samples[self.field_samples[field_ssfr_col]<sfr_cut_1]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_passive.groupby(pd.cut(sn_passive[i],
                                                     bins=mbins))[[i,weight_col_SN]]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_passive.groupby( pd.cut(field_passive[i_f],
                                                            bins=mbins))[[i_f,weight_col_field]]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g[weight_col_SN].sum()/g2[weight_col_field].sum())+self.rate_corr) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)
        if not savename:
            savename=self.config['rates_root']+'data/mcd_rates.h5'
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_passive')
        self.sampled_passive_rates = iter_df

        #moderately starforming
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        sn_moderate = self.sn_samples[(self.sn_samples[sn_ssfr_col]>=sfr_cut_1)&(self.sn_samples[sn_ssfr_col]<sfr_cut_2)]
        field_moderate = self.field_samples[(self.field_samples[field_ssfr_col]>=sfr_cut_1)&(self.field_samples[field_ssfr_col]<sfr_cut_2)]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_moderate.groupby(pd.cut(sn_moderate[i],
                                                     bins=mbins))[[i,weight_col_SN]]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_moderate.groupby( pd.cut(field_moderate[i_f],
                                                            bins=mbins))[[i_f,weight_col_field]]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g[weight_col_SN].sum()/g2[weight_col_field].sum())+self.rate_corr) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)

        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_moderate')
        self.sampled_moderate_rates = iter_df

        #highly starforming
        iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)
        sn_high = self.sn_samples[self.sn_samples[sn_ssfr_col]>=sfr_cut_2]
        field_high = self.field_samples[self.field_samples[field_ssfr_col]>=sfr_cut_2]
        with progressbar.ProgressBar(max_value = n_samples) as bar:
            for i in range(0,n_samples):
                snmassgroups =sn_high.groupby(pd.cut(sn_high[i],
                                                     bins=mbins))[[i,weight_col_SN]]
                i_f = np.random.randint(0,100)
                fieldmassgroups = field_high.groupby( pd.cut(field_high[i_f],
                                                            bins=mbins))[[i_f,weight_col_field]]
                xs = []
                ys = []

                for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                    if g.size >0 and g2.size>0:
                        xs.append(n.mid)

                        ys.append(np.log10(g[weight_col_SN].sum()/g2[weight_col_field].sum())+self.rate_corr) # We want a per-year rate.

                xs = np.array(xs)
                ys = np.array(ys)
                entry = pd.Series(ys,index=xs)
                iter_df.loc[entry.index,i] = entry
                bar.update(i)

        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_high')
        self.sampled_high_rates = iter_df

    def load_sampled_rates(self,fn,ext='mass'):
        if ext:

            df = pd.read_hdf(fn,key='bootstrap_samples_%s'%ext)
            setattr(self,'sampled_rates_%s'%ext,df)
        else:
            df = pd.read_hdf(fn,key='bootstrap_samples')
            setattr(self,'sampled_rates',df)
        return df
    def fit_line(self,df,xmin=8,xmax=11,seed=123456,n_iter=4E3,dispersion=False,**kwargs):
        if dispersion:

            model = stan_utility.compile_model(self.root_dir+'models/fit_yline_hetero_scatter.stan')
        else:
            model = stan_utility.compile_model(self.root_dir+'models/fit_yline_hetero.stan')
        x_model = np.linspace(xmin,xmax,100)
        x_obs = np.array(df.loc[xmin:xmax].index)
        y_obs = df.mean(axis=1).loc[xmin:xmax].values
        y_err = df.std(axis=1).loc[xmin:xmax].values

        data = dict(N = len(x_obs),
                    x_obs = x_obs,
                    y_obs = y_obs,
                    #sigma_x=np.array(xerr[:-2]),
                    sigma=y_err,
                    N_model=100,
                   x_model=x_model)
        fit = model.sampling(data=data, seed=seed, iter=int(n_iter))
        return fit



    def plot_fit_mass(self,fit,rate,mmin=8,mmax=11,f=None,ax=None,label_text=None,line_only=False,**kwargs):
        if not f:
            f,ax = plt.subplots(figsize=(12,7))

        chain = fit.extract()

    # Plot the points from above as a comparison
        x_model = np.linspace(mmin,mmax,100)
        if not line_only:
            for counter,c in enumerate(rate.columns):
                label=None

                ax.scatter(rate.index,rate[c],marker='o',
                               alpha=0.05,s=10,label=None)
                ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(0.125))
                ax.tick_params(which='both',right=True,top=True,direction='in',labelsize=16)
                ax.set_xlabel('Stellar Mass $\log (M_*/M_{\odot})$',size=20)
                ax.set_ylabel('$\log (N$ (SN hosts) / $N$ (Field Galaxies) )',size=20)
            for i in rate.index:
                ax.errorbar(i,rate.loc[i].mean(),xerr=(rate.index[1]-rate.index[0])/2,
                                ,marker='D',alpha=0.5,markersize=2,mew=0.5,mec='w')

        level = 95

        ax.fill_between(x_model,
                        np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                        np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                        alpha=0.2)

        level = 68
        ax.fill_between(x_model,
                        np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                        np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                        alpha=0.3)
        if label_text:
            label= label_text+': $dR/dM_* = %.2f +/- %.2f$'%(np.median(chain['slope']),np.std(chain['slope']))
        else:
            label= '$dR/dM_* = %.2f +/- %.2f$'%(np.median(chain['slope']),np.std(chain['slope']))
        ax.plot(x_model,
                        np.percentile(chain['line'], 50, axis=0 ),
                        alpha=1,linestyle='-',linewidth=1,label=label)

        leg =ax.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.savefig(self.root_dir +'figs/rate_vs_mass_slopes_stanfit_test.png')
        return f,ax
    def plot_fit_sfr(self,fit,sfrmin=-3,sfrmax=2):
        fmbinlog,axmbinlog = plt.subplots(figsize=(12,7))
        chain = fit.extract()

    # Plot the points from above as a comparison
        x_model = np.linspace(sfrmin,sfrmax,100)
        for counter,c in enumerate(self.sampled_rates_sfr.columns):
            label=None
            if counter == 0:
                label='Observations'
            axmbinlog.scatter(self.sampled_rates_sfr.index,self.sampled_rates_sfr[c],color='g',marker='o',
                           alpha=0.05,s=10,label=label)
            axmbinlog.xaxis.set_minor_locator(MultipleLocator(0.25))
            axmbinlog.yaxis.set_minor_locator(MultipleLocator(0.125))
            axmbinlog.tick_params(which='both',right=True,top=True,direction='in',labelsize=16)
            axmbinlog.set_xlabel('$\log$ (SFR / $M_{\odot}$ yr$^{-1}$)',size=20)
            axmbinlog.set_ylabel('$\log (N$ (SN hosts) / $N$ (Field Galaxies) )',size=20)
        for i in self.sampled_rates_sfr.index:
            axmbinlog.errorbar(i,self.sampled_rates_sfr.loc[i].mean(),xerr=(self.sampled_rates_sfr.index[1]-self.sampled_rates_sfr.index[0])/2,
                            color='g',marker='D',alpha=0.5,markersize=2,mew=0.5,mec='w')

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
                        color='b',alpha=1,linestyle='-',linewidth=1,label='$dR/dSFR = %.2f$'%np.median(chain['slope']))

        leg =axmbinlog.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.savefig(self.root_dir +'figs/rate_vs_sfr_slopes_stanfit_test.png')

    def plot_fit_split_SFR(self,fits,rates,f=None,ax=None):
        if not f:
            f,ax = plt.subplots(figsize=(12,7))
        palette =itertools.cycle(sns.color_palette(palette='husl',n_colors=8))
    # Plot the points from above as a comparison
        names=['Passive','Moderate','High']
        for j in range(len(fits)):
            colour=next(palette)
            chain = fits[j].extract()
            x_model = np.linspace(6.5,11,100)
            #x_model = np.linspace(8.125,11,100)
            for counter,c in enumerate(rates[j].columns):
                label=None
                if counter == 0:
                    label=names[j]
                ax.scatter(rates[j].index,rates[j][c],color=colour,marker='o',
                               alpha=0.05,s=10,label=label)
                ax.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax.yaxis.set_minor_locator(MultipleLocator(0.125))
                ax.tick_params(which='both',right=True,top=True,direction='in',labelsize=16)
                ax.set_xlabel('Stellar Mass $\log (M_*/M_{\odot})$',size=20)
                ax.set_ylabel('$\log (N$ (SN hosts) / $N$ (Field Galaxies) )',size=20)
            for i in rates[j].index:
                ax.errorbar(i,rates[j].loc[i].mean(),xerr=0.125,color=colour,marker='D',
                               alpha=0.5,markersize=2,mew=0.5,mec='w')

            level = 95

            ax.fill_between(x_model,
                            np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                            np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                            color=colour,alpha=0.2)

            level = 68
            ax.fill_between(x_model,
                            np.percentile(chain['line'], 50 - 0.5*level, axis=0 ),
                            np.percentile(chain['line'], 50 + 0.5*level, axis=0 ),
                            color=colour,alpha=0.3)

            ax.plot(x_model,
                            np.percentile(chain['line'], 50, axis=0 ),
                            color=colour,alpha=1,linestyle='-',linewidth=1,label='$dR/dM_* = %.2f$'%np.median(chain['slope']))
            leg =ax.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
        #plt.savefig(r.root_dir +'figs/rate_vs_mass_slopes_stanfit.png')
        return f,ax
