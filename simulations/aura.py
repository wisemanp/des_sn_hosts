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
from .utils.gal_functions import schechter, single_schechter, double_schechter, ozdes_efficiency, interpolate_zdf
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
    def __init__(self,conf_path,cosmo='default'):
        '''

        :param fn:
        :type fn:
        :param cosmo:
        :type cosmo:
        '''
        self.config = self._get_config(conf_path)
        root_dir = self.config['config']['root_dir']
        if root_dir[0]=='$':
            self.root_dir = os.environ.get(root_dir[1:])
        else:
            self.root_dir = root_dir
        self.fig_dir = os.path.join(self.root_dir,'figs/')
        self.eff_dir = self.config['config']['efficiency_dir']
        self.flux_df = self._load_flux_df(self.config['hostlib_fn'])
        self._calculate_absolute_rates()
        self._make_multi_index()
        if cosmo=='default':
            self.cosmo = FlatLambdaCDM(70,0.3)
        else:
            self.cosmo=cosmo
        self._get_funcs()

    def _get_config(self,conf_path):
        with open(conf_path,'r') as f:
            return yload(f)
    def _load_flux_df(self,fn):
        return pd.read_hdf(fn)

    def _calculate_absolute_rates(self):
        '''For each simulated galaxy, we calculate the expected rate of SNe Ia by multplying it with the stellar mass density at that stellar mass
        TODO: Currently this is for the SMF at z=0.5 but should update it to vary with redshift'''

        #phi_star_1=3.96E-3
        #alpha_1=-0.35
        #phi_star_2=0.79E-3
        #alpha_2=-1.47
        #mstar = 10.66
        #self.flux_df['phi']= self.flux_df['mass'].apply(lambda x: double_schechter(np.log10(x),mstar,phi_star_1,alpha_1,phi_star_2,alpha_2))
        self.flux_df['phi']= self.flux_df[['z','mass']].apply(lambda x: schechter(x[0],np.log10(x[1])),axis=1)
        self.flux_df['N_x1_lo'] = self.flux_df['pred_rate_x1_lo']*self.flux_df['phi']
        self.flux_df['N_x1_hi'] = self.flux_df['pred_rate_x1_hi']*self.flux_df['phi']
        self.flux_df['N_total'] = self.flux_df['pred_rate_total']*self.flux_df['phi']

    def _make_multi_index(self):
        ''' Convert the DataFrame to have a multi-index structure to enable us to easily select only one galaxy for each sample of [redshift, mass, Av]'''
        z_str = self.flux_df['z'].apply(lambda x: '%.3f'%x)
        mass_str = self.flux_df['mass'].apply(lambda x: '%.2f'%x)
        Av_str = self.flux_df['Av'].apply(lambda x: '%.5f'%x)
        # We now have three levels to the index: z, mass, Av. For any given z and mass, the stellar populations are identical at any Av, but the output fluxes and colours are not.
        self.multi_df = self.flux_df.set_index([z_str,mass_str,Av_str,])

    def _get_z_dist(self,z_vals,n=25000,frac_low_z=0.2,zbins=[]):
        '''

        :param z_vals: an array of redshifts that will have the same distribution that you want the simulation to have. This can be observed or simulated data.
        :type z_vals:
        :return:
        :rtype:
        '''
        if len(zbins)==0:
            zbins=[0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625,
                  0.675,0.725,0.775,0.825]
        counts, bins = np.histogram(z_vals,
                                    bins=zbins)
        norm_counts = counts / np.sum(counts)
        norm_counts = counts / np.sum(counts)
        n_samples_arr = norm_counts * n
        n_samples_arr = np.concatenate([[n*frac_low_z], n_samples_arr])
        return n_samples_arr

    def _get_funcs(self):
        self.rv_func = getattr(self, self.config['SN_rv_model']['model'])
        self.host_Av_func = getattr(self, self.config['Host_Av_model']['model'])
        self.E_func = getattr(self, self.config['SN_E_model']['model'])
        self.colour_func = getattr(self, self.config['SN_colour_model']['model'])
        self.x1_func = getattr(self, self.config['x1_model']['model'])
        self.mb_func = getattr(self, self.config['mB_model']['model'])
        self.save_string = self.rv_func.__name__ + '_'+\
                self.host_Av_func.__name__ + '_' +\
                self.E_func.__name__ + '_' +\
                self.colour_func.__name__ + '_' +\
                self.x1_func.__name__ + '_' +\
                self.mb_func.__name__
    def sample_SNe(self,z_arr,n_samples_arr,save_df=True,savepath='default'):
        self.sim_df = pd.DataFrame()

        for z,n in zip(z_arr,n_samples_arr):
            self.sim_df = self.sim_df.append(self._sample_SNe_z(z,n))
        if save_df:
            if savepath=='default':
                savepath = self.root_dir +'/sims/SNe/'+ self.save_string +'_SN_sim.h5'
            self.savepath=savepath
            self.sim_df.to_hdf(self.savepath,key='sim')

    def _sample_SNe_z(self,z,n_samples):
        #print('*** Sampling %i SNe at redshift %.2f'%(n_samples,z))
        if n_samples == 0:
            return pd.DataFrame(columns=self.sim_df.columns)
        args = {}

        args['n'] = int(n_samples)
        args['distmod'] = self.cosmo.distmod(z).value

        z_df = self.multi_df.loc['%.3f' % z].copy()

        z_df['N_total'].replace(0., np.NaN, inplace=True)
        z_df.dropna(subset=['N_total'],inplace=True)

        z_df['N_SN_float'] = z_df['N_total'] / z_df['N_total'].min()  # Normalise the number of SNe so that the most improbable galaxy gets 1

        z_df['N_SN_int'] = z_df.loc[:,'N_SN_float'].astype(int)

        resampled_df = pd.DataFrame()
        marr= np.logspace(6,11.6,100)
        for av in z_df.Av.unique():
            av_df =z_df.loc[idx[:, '%.5f'%av, :]]
            #print(av_df)
            av_df = interpolate_zdf(av_df,marr)
            resampled_df = resampled_df.append(av_df)
        #print(resampled_df.columns)
        Av_str = resampled_df['Av'].apply(lambda x: '%.5f'%x)
        mass_str = resampled_df['mass'].apply(lambda x: '%.2f'%x)
        new_zdf = resampled_df.set_index([mass_str,Av_str])
        m_inds = ['%.2f' % m for m in new_zdf['mass'].unique()]

        m_rates = []
        m_rates_float = []
        for m in m_inds:
            m_df = new_zdf.loc[m]
            mav_inds = (m, '%.5f' % (m_df.Av.unique()[0]))
            #print(new_zdf.loc[mav_inds,'N_SN_int'])
            m_rates.append(new_zdf.loc[mav_inds,'N_SN_int'])
            m_rates_float.append(new_zdf.loc[m,'N_SN_float'])


        m_samples = np.random.choice(m_inds, p=m_rates / np.sum(m_rates), size=int(n_samples))
        # Now we have our masses, but each one needs some reddening. For now, we just select Av at random from the possible Avs in each galaxy
        # The stellar population arrays are identical no matter what the Av is.
        m_av0_samples = [(m, '%.5f' % (np.random.choice(new_zdf.loc[m].Av.values))) for m in m_samples]
        new_zdf['SN_ages'] = [age_grid for i in range(len(new_zdf))]
        new_zdf['SN_age_dist'] = [np.zeros(len(age_grid)) for i in range(len(new_zdf))]

        age_dists = []
        for n,g in z_df.groupby(pd.cut(z_df['mass'],bins=marr)):
            age_df = pd.DataFrame(index=age_grid_index)
            if len(g)>0:
                g_Av_0 =  g.loc[idx[:, '0.00000', :]]
                for k in g_Av_0.index:
                    sub_gb = g_Av_0.loc[k]
                    age_inds = ['%.4f'%a for a in sub_gb['SN_ages']]
                    age_df.loc[age_inds,'%.2f'%(g_Av_0['mass'].mean())] = sub_gb['SN_age_dist'].values
                age_df.fillna(0,inplace=True)
                for av in g.Av.unique():
                    age_dists.append(np.nanmean(age_df,axis=1))
            else:
                pass

        new_zdf['SN_age_dist']=age_dists
        # Now we sample from our galaxy mass distribution, given the expected rate of SNe at each galaxy mass
        gals_df = new_zdf.loc[m_av0_samples,['z','mass','ssfr','m_g','m_r','m_i','m_z','U', 'B', 'V', 'R', 'I','U_R','mean_age','Av','pred_rate_total']]

        sn_ages = [np.random.choice(new_zdf.loc[i,'SN_ages'],p=new_zdf.loc[i,'SN_age_dist']/new_zdf.loc[i,'SN_age_dist'].sum()) for i in m_av0_samples]
        gals_df['SN_age'] = np.array(sn_ages)
        args['Av_grid'] = new_zdf.Av.unique()
        args['mass'] = gals_df.mass.values
        args['ssfr'] = gals_df.ssfr.values
        args['sfr'] = args['mass']*args['ssfr']
        args['mean_ages'] = gals_df.mean_age.values
        args['SN_age'] = np.array(sn_ages)
        args['rv'] = self.rv_func(args,self.config['SN_rv_model']['params'])
        if  self.config['SN_E_model']['model'] in ['E_calc','E_from_host_random']:
            args['host_Av'] = self.host_Av_func(args, self.config['Host_Av_model']['params'])
            args['E'] = self.E_func(args, self.config['SN_E_model']['params'])
        else:
            args['E'] = self.E_func(args, self.config['SN_E_model']['params'])
            args['host_Av'] = self.host_Av_func(args, self.config['Host_Av_model']['params'])


        args['host_Av'] = self.host_Av_func(args,self.config['Host_Av_model']['params'])
        m_av_samples_inds = [[m_samples[i],'%.5f'%(args['host_Av'][i])] for i in range(len(args['host_Av']))]

        args['U-R'] = gals_df['U'].values - gals_df['R'].values #gal_df['U_R'].values
        for band in ['g','r','i','z']:
            args['m_%s'%band] = gals_df['m_%s'%band].values-1
        mean_eff_func,std_eff_func = ozdes_efficiency(self.eff_dir)
        spec_eff = mean_eff_func(args['m_r'])
        spec_eff_std = std_eff_func(args['m_r'])
        effs = np.clip(np.random.normal(spec_eff,spec_eff_std),a_min=0,a_max=1)
        args['eff_mask'] = [np.random.choice([0,1],p=[1-effs[i],effs[i]]) for i in range(len(effs))]

        args = self.colour_func(args,self.config['SN_colour_model']['params'])
        args = self.x1_func(args,self.config['x1_model']['params'])
        args['mB'],args['beta_SN'] = self.mb_func(args,self.config['mB_model']['params'])
        args['mB_err'] =[np.max([0.025,np.random.normal(10**(0.395*(args['mB'][i]-1.5) - 10)+0.03,np.max([0.003,0.003*(args['mB'][i]-20)]))])
                         for i in range(len(args['mB']))]

        args['c_err'] = [np.max([0.02,np.random.normal((0.675*args['mB_err'][i] +0.005),0.003)])
                         for i in range(len(args['mB']))]

        args['c_noise'] =norm(0,args['c_err']).rvs(size=len(args['c']))
        args['c'] = args['c'] + args['c_noise']

        args['x1_err'] = [np.max([0.08,np.random.normal((14*args['mB_err'][i] -0.25 ),0.05)])
                         for i in range(len(args['mB']))]
        args['x1_noise'] =norm(0,args['x1_err']).rvs(size=len(args['x1']))
        args['x1_int'] = args['x1'].copy()
        args['x1'] = args['x1'] + args['x1_noise']
        #C = np.cov([args['mB'],args['x1'],args['c']])

        args['cov_mB_x1'],args['cov_mB_c'],args['cov_x1_c'] = 0,0,0 #Set covariance off-diagonal terms to 0 for now #C[0,1],C[0,2],C[1,2]
        self.args = args
        args['distmod'] = np.ones_like(args['c'])*args['distmod']
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
