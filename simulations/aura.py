import numpy as np
import pandas as pd
from astropy.table import Table
import os
from yaml import safe_load as yload
import scipy.stats as stats
import sys
import pickle
import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.cosmology import FlatLambdaCDM

from .models.sn_model import SN_Model
from .utils.gal_functions import schechter, single_schechter, double_schechter
from .utils.plotter import *

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import itertools
from tqdm import tqdm


aura_dir = os.environ['AURA_DIR']

class Sim(SN_Model):
    def __init__(self,conf_path,cosmo='default'):
        '''

        :param fn:
        :type fn:
        :param cosmo:
        :type cosmo:
        '''
        self.config = self._get_config(conf_path)
        self.flux_df = self._load_flux_df(self.config['hostlib_fn'])
        self._calculate_absolute_rates()
        self._make_multi_index()
        if cosmo=='default':
            self.cosmo = FlatLambdaCDM(70,0.3)
        else:
            self.cosmo=cosmo

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
        z_str = self.flux_df['z'].apply(lambda x: '%.2f'%x)
        mass_str = self.flux_df['mass'].apply(lambda x: '%.2f'%x)
        Av_str = self.flux_df['Av'].apply(lambda x: '%.5f'%x)
        # We now have three levels to the index: z, mass, Av. For any given z and mass, the stellar populations are identical at any Av, but the output fluxes and colours are not.
        self.multi_df = self.flux_df.set_index([z_str,mass_str,Av_str,])

    def _get_z_dist(self,z_vals):
        '''

        :param z_vals: an array of redshifts that will have the same distribution that you want the simulation to have. This can be observed or simulated data.
        :type z_vals:
        :return:
        :rtype:
        '''
        counts, bins = np.histogram(z_vals,
                                    bins=[0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625,
                                          0.675])
        norm_counts = counts / np.sum(counts)
        norm_counts = counts / np.sum(counts)
        n_samples_arr = norm_counts * 25000
        n_samples_arr = np.concatenate([[5000], n_samples_arr])
        return n_samples_arr

    def sample_SNe(self,z_arr,n_samples_arr):
        self.sim_df = pd.DataFrame()
        for z,n in zip(z_arr,n_samples_arr):
            self.sim_df = self.sim_df.append(self._sample_SNe_z(z,n))

    def _sample_SNe_z(self,z,n_samples):
        args = {}
        args['distmod'] = self.cosmo.distmod(z).value
        z_df = self.multi_df.loc['%.2f' % z]
        z_df['N_total'].replace(0., np.NaN, inplace=True)
        z_df = z_df.dropna(subset=['N_total'])
        z_df['N_SN_float'] = z_df['N_total'] / z_df['N_total'].min()  # Normalise the number of SNe so that the most improbable galaxy gets 1
        z_df['N_SN_int'] = z_df['N_SN_float'].astype(int)
        # Now we set up some index arrays so that we can sample masses properly
        m_inds = ['%.2f' % m for m in z_df['mass'].unique()]
        m_rates = []
        for m in m_inds:
            m_df = z_df.loc[m]
            mav_inds = (m, '%.5f' % (m_df.Av.unique()[0]))
            m_rates.append(z_df.loc[mav_inds]['N_SN_int'])

        # Now we sample from our galaxy mass distribution, given the expected rate of SNe at each galaxy mass
        m_samples = np.random.choice(m_inds, p=m_rates / np.sum(m_rates), size=int(n_samples))
        # Now we have our masses, but each one needs some reddening. For now, we just select Av at random from the possible Avs in each galaxy
        # The stellar population arrays are identical no matter what the Av is.
        m_av0_samples = [(m, '%.5f' % (np.random.choice(z_df.loc[m].Av.values))) for m in m_samples]

        args['Av_grid'] = z_df.Av.unique()
        args['mass'] = z_df.loc[m_av0_samples].mass.values
        args['age'] = z_df.loc[m_av0_samples].mean_age.values

        rv_func = getattr(self, self.config['SN_rv_model']['model'])
        args['rv'] = rv_func(args,self.config['SN_rv_model']['params'])

        host_Av_func = getattr(self,self.config['Host_Av_model']['model'])
        E_func = getattr(self, self.config['SN_E_model']['model'])
        args['E'] = E_func(args, self.config['SN_E_model']['params'])
        args['host_Av'] = host_Av_func(args,self.config['Host_Av_model']['params'])
        m_av_samples_inds = [[m_av0_samples[i],'%.5f'%(args['host_Av'][i])] for i in range(len(args['host_Av']))]
        gal_df = z_df.loc[m_av_samples_inds]
        args['U-R'] = gal_df['U-R']
        args['mean_ages'] = gal_df['mean_age']



        colour_func = getattr(self,self.config['SN_colour_model']['model'])
        args['c'] = colour_func(args,self.config['SN_colour_model']['params'])

        x1_func = getattr(self,self.config['x1_model']['model'])
        args = x1_func(args,self.config['x1_model']['params'])

        mb_func = getattr(self,self.config['mB_model']['model'])
        args['mB'] = mb_func(args,self.config['mb_model']['params'])
        args['mB_err'] =np.max([0.03,np.random.normal(10**(0.4*(args['mB']-1.5) - 10)+0.02,0.03)])
        z_sim_df = pd.DataFrame(args)
        z_sim_df['z'] = z
        return z_sim_df



    def _sample_SNe(self,z,dust,M0,alpha,sigma_alpha,beta,sigma_beta,sigma_int,gamma_m, gamma_l,mass_step_loc,age_step_loc,method,which,n_samples,beta_young=2,beta_old=3,alpha_young=0.1,alpha_old=0.2):
        ''' Function that takes a sample of SNe given a set of input parameters. This is the 'hidden' working function; the user interacts via sample_SNe()'''
        print(z)
        distmod = self.cosmo.distmod(z).value
        z_df = self.multi_df.loc['%.2f'%z]
        z_df['N_%s'%which].replace(0.,np.NaN,inplace=True)
        z_df = z_df.dropna(subset=['N_%s'%which])

        #Normalise the number of SNe so that the most improbable galaxy gets 1
        z_df['N_SN_float'] =z_df['N_%s'%which]/z_df['N_%s'%which].min()
        if which=='total':
            rate_factor=1
        else:
            # If we are splitting the rates up by SN age, make sure we get the normalisation right
            print("Splitting by age")
            z_df['N_total'].replace(0.,np.NaN,inplace=True)
            z_df = z_df.dropna(subset=['N_total'])
            ntot = np.sum(z_df['N_total'])
            rate_factor = np.sum(z_df['N_%s'%which])/ntot
            print("There should be %.2f times as many %s SNe"%(rate_factor,which))
        z_df['N_SN_int']= z_df['N_SN_float'].astype(int)

        # Now we set up some index arrays so that we can sample masses properly
        m_inds = ['%.2f'%m for m in z_df['mass'].unique()]
        m_rates = []
        for m in m_inds:
            m_df = z_df.loc[m]
            mav_inds =(m,'%.5f'%(m_df.Av.unique()[0]))
            m_rates.append(z_df.loc[mav_inds]['N_SN_int'])

        # Now we sample from our galaxy mass distribution, given the expected rate of SNe at each galaxy mass
        m_samples = np.random.choice(m_inds,p=m_rates/np.sum(m_rates),size=int(n_samples*rate_factor))

        # Set up some results lists
        m_av_samples_inds = []
        Es = []
        rvs_sn =[]
        Eslo = []
        Eshi = []
        n_mhi =0
        n_mlo=0
        # Now we have our masses, but each one needs some reddening. For now, we just select Av at random from the possible Avs in each galaxy
        # The stellar population arrays are identical no matter what the Av is.
        m_av0_samples = [(m,'%.5f'%(np.random.choice(z_df.loc[m].Av.values))) for m in m_samples]
        E_grid = np.linspace(0.001,3,1000)
        Av_grid = z_df.Av.unique()
        if dust =='dist':
            # We're going to base our dust off the SN dust distribution and use that to guess what the host dust looks like.
            m_samples_float =z_df.loc[m_av0_samples].mass.values
            age_samples_float = z_df.loc[m_av0_samples].mean_age.values  # This is the mass weighted mean age of the stellar population
            avs_mlo,avs_mhi = [],[]


            for n,(m,age) in enumerate(zip(m_samples_float,age_samples_float)):
                # Here we are going to draw our SN dust and our host dust.
                # We draw E(B-V) from an exponential distribution 1/tauE * exp(-E/tauE). In some models tauE is different depending on the host mass or age.
                # We do this by setting up a probability distribution PEs using the function P_Edust(), then selecting from it randomly
                # Then we need to chose a dust law slope, e.g. Rv. This is model dependent.
                # After sampling a SN Rv, we move on to the host dust. Our "Av_grid" is the different Avs that have been used to extinguish the fluxes of the galaxy spectra
                # We sample our host Av according to the E(B-V) of the SN. We sample from a Gaussian centered at the Av given by the SN E(B-V)/Rv_host where Rv_host is model dependent


                # We could split our dust properties based on the host stellar population age
                if method[:8]=='rv_age_g':
                    if age<=3500:
                        PEs = P_Edust(E_grid,TauE=0.12)
                        E=np.random.choice(E_grid,p=PEs/np.sum(PEs))
                        Es.append(E)
                        Eslo.append(E)

                        # BS20 use set distributions of Rv in low and high mass hosts
                        rv = np.max([np.random.normal(2.25,1.3),0.5])
                        # sample the host Av according to the SN E(B-V) but a standard MW dust law with mean Rv=3.1
                        pAv = norm(E/3.1,0.5).pdf(Av_grid)

                        av_mlo=np.random.choice(Av_grid,p=pAv/np.sum(pAv),)
                        avs_mlo.append(av_mlo)
                        m_av_samples_inds.append((m_samples[n],'%.5f'%av_mlo))
                        n_mlo+=1
                    else:
                        PEs = P_Edust(E_grid,TauE=0.15)
                        E=np.random.choice(E_grid,p=PEs/np.sum(PEs))
                        Es.append(E)
                        Eshi.append(E)

                        # BS20 use set distributions of Rv in low and high mass hosts
                        rv = np.max([np.random.normal(1.5,1.3),0.5])
                        # sample the host Av according to the SN E(B-V) but a standard dust law with mean Rv=2.6
                        pAv = norm(E/2.6,0.5).pdf(Av_grid)
                        av_mhi=np.random.choice(Av_grid,p=pAv/np.sum(pAv),)
                        avs_mhi.append(av_mhi)
                        m_av_samples_inds.append((m_samples[n],'%.5f'%av_mhi))
                        n_mhi+=1
                # Or we could split our dust properties based on the host stellar mass
                else:
                    if m<=1E+10:

                        if method[:4]=='BS20':
                            PEs = P_Edust(E_grid,TauE=0.12)
                        else:
                            PEs = P_Edust(E_grid,TauE=0.1)
                        E=np.random.choice(E_grid,p=PEs/np.sum(PEs))
                        Es.append(E)
                        Eslo.append(E)
                        if method[:4]=='BS20':
                            # BS20 use set distributions of Rv in low and high mass hosts
                            # We can fiddle with the mean of this distribution
                            rv = np.max([np.random.normal(2.75,1.3),0.5])
                            # sample the host Av according to the SN E(B-V) but a standard MW dust law with mean Rv=3.1
                            pAv = norm(E/3.1,0.5).pdf(Av_grid)
                        elif method[:3]=='SFR':
                            # In case we want to split the dust on host SFR
                            pass
                        else:
                            # assume dust law is independent of galaxy mass
                            rv = np.max([np.random.normal(2.65,1.3),1.2])
                            pAv = norm(E/rv,0.5).pdf(Av_grid)
                        av_mlo=np.random.choice(Av_grid,p=pAv/np.sum(pAv),)
                        avs_mlo.append(av_mlo)
                        m_av_samples_inds.append((m_samples[n],'%.5f'%av_mlo))
                        n_mlo+=1
                    else:
                        if method[:4]=='BS20':
                            PEs = P_Edust(E_grid,TauE=0.15)
                        else:
                            PEs = P_Edust(E_grid,TauE=0.1)
                        E=np.random.choice(E_grid,p=PEs/np.sum(PEs))
                        Es.append(E)
                        Eshi.append(E)
                        if method[:4]=='BS20':
                            # BS20 use set distributions of Rv in low and high mass hosts
                            rv = np.max([np.random.normal(1.5,1.3),0.5])
                            # sample the host Av according to the SN E(B-V) but a standard dust law with mean Rv=2.6
                            pAv = norm(E/2.6,0.5).pdf(Av_grid)
                        elif method[:3]=='SFR':
                            pass
                        else:
                            # assume dust law is independent of galaxy mass
                            rv = np.max([np.random.normal(2.65,1.3),1.2])
                            pAv = norm(E/rv,0.5).pdf(Av_grid)
                        av_mhi=np.random.choice(Av_grid,p=pAv/np.sum(pAv),)
                        avs_mhi.append(av_mhi)
                        m_av_samples_inds.append((m_samples[n],'%.5f'%av_mhi))
                        n_mhi+=1
                rvs_sn.append(rv)
            #P_avs_lo = P_Edust(z_df.Av.unique()/2.75,TauE =0.12)
            #P_avs_hi = P_Edust(z_df.Av.unique()/1.5,TauE =0.15)
            #avs_mlo = np.random.choice(z_df.Av.unique(),p=P_avs_lo/np.sum(P_avs_lo),size=len(m_samples_float[m_samples_float<=1E+10]))
            #avs_mhi = np.random.choice(z_df.Av.unique(),p=P_avs_hi/np.sum(P_avs_hi),size=len(m_samples_float[m_samples_float>1E+10]))
            #avs_mlo_SBL18 = np.random.normal(np.ones_like(avs_mlo)*av_means_mlo,av_sigma(np.log10(m_samples_float[m_samples_float<=1E+10])))
            #avs_mhi_SBL18 = np.random.normal(av_means_mhi(np.log10(m_samples_float[m_samples_float>1E+10])),av_sigma(np.log10(m_samples_float[m_samples_float>1E+10])))
            gals_df = z_df.loc[m_av_samples_inds]
        else:
            gals_df = z_df.loc[m_av0_samples]

        # I think this is not used any more as I do put Rv into the data frame
        if 'Rv' not in gals_df.columns:
            if method[:4]=='BS20':
                z_df['Rv'] = z_df['mass'].apply(choose_rv)
            else:
                z_df['Rv'] = rv
       # Now we have a sample of SNe and their hosts, with SN and host dust already sampled. We now go on to sample the SN age and light curve parameters
        for counter,ind in tqdm(enumerate(gals_df.index)):
            row = z_df.loc[ind]

            # First let's sample a SN progenitor age, from the SN age probability density according to the star formation history of the galaxy and the DTD
            age = np.random.choice(row['SN_ages'],p = row['SN_age_dist'].fillna(0)/row['SN_age_dist'].fillna(0).sum())
            self.SN_ages.append(age)
            self.masses.append(row['mass'])
            self.U_Rs.append(row['U_R'])
            self.Avs.append(row['Av'])
            self.Rvs.append(row['Rv'])
            self.Rv_SNe.append(rvs_sn[counter])
            self.zs.append(row['z'])

            if dust =="host":
                # If instead we get the SN reddening from the host dust, then we use the randomly sampled host Av and a given Rv to sample the E(B-V)
                Es.append(row['Av']/row['Rv'])
                self.E_SNs.append(row['Av']/row['Rv'])
            else:
                self.E_SNs.append(Es[counter])
            self.mean_ages.append(row['mean_age']) # this is the mass weighted mean stellar population age

            # now let's sample an x1!
            if method[:8]=='fixed_x1' or method[:13]=='BS20_fixed_x1':
                x1grid = np.linspace(-5,5,100000)
                px1s = asymmetric_gaussian(x1grid,DES5yrmu,DES5yrsigminus,DES5yrsigplus)
                x1_sim = np.random.choice(x1grid,p=px1s/np.sum(px1s))
            else:

                if which=='total':   # Use this unless using separate DTD for low and high stretch objects
                    if age > age_step_loc:
                        # If the progenitor is old, we sample from two Gaussians, W21_old and W21_young, which we can tweak to fit the data.
                        x1s = {'old': W21_old.rvs(),'young':W21_young.rvs()}
                        # Now we choose whether we sample from the young or old Gaussian, with some weights (which we can tweak)
                        prog_age_choice = np.random.choice(['old','young'],p=[0.6,0.4])
                        x1_sim =x1s[prog_age_choice]   #+np.random.normal(0,0.35)
                    else:
                        # If the progenitor is young, we sample from the young Gaussian
                        prog_age_choice = 'young'
                        x1_sim =W21_young.rvs()   #+np.random.normal(0,0.35)
                elif which=='x1_lo':
                    x1_sim = W21_old.rvs()   #+np.random.normal(0,0.35)
                elif which=='x1_hi':
                    x1_sim = W21_young.rvs()   #+np.random.normal(0,0.35)

            self.x1s.append(x1_sim)

            # Now let's sample some colours and betas!
            if method[:8] =='two_beta':
                # We could imagine that SNe have two betas; one for old and one for young progenitors
                if age >age_step_loc:
                    cint_sn = np.random.choice([self.cint_old.rvs(),self.cint_young.rvs()],p=[0.5,0.5])
                    beta_sim= np.random.choice([np.random.normal(beta_old,sigma_beta),
                                                np.random.normal(beta_young,sigma_beta)],
                                               p=[0.5,0.5])
                    self.betas.append(beta_sim)
                else:
                    cint_sn = self.cint_young.rvs()
                    beta_sim= np.random.normal(beta_young,sigma_beta)
                    self.betas.append(beta_sim)
            else:
                # in most cases, we sample intrinsic colour from a single Gaussian, which we can set as an input
                cint_sn = self.cint.rvs()
                # we also draw beta from a normal distribtion
                beta_sim= np.random.normal(beta,sigma_beta)
                self.betas.append(beta_sim)

            # we could also have two alphas for different age progenitors which we sample according to whether we sampled from the old your young x1 distributions
            if method[:9] =='two_alpha':
                if age >age_step_loc:
                    alphas = {'old':norm(alpha_old,sigma_alpha).rvs() ,'young':norm(alpha_young,sigma_alpha).rvs()}
                    alpha_sim= alphas[prog_age_choice]
                    self.alphas.append(alpha_sim)
                else:
                    alpha_sim= np.random.normal(alpha_young,sigma_alpha)
                    self.alphas.append(alpha_sim)
            else:
                alpha_sim = np.random.normal(alpha,sigma_alpha)

            # Our observed colour is our SN colour plus some dust reddening
            c_sim = cint_sn +Es[counter]#+np.random.normal(0,0.04)
            self.cs.append(c_sim)
            self.cints.append(cint_sn)

            # calculate the observed mB depending on our model!
            if method =='linear_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*c_sim - alpha*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) #np.random.normal(beta,sigma_beta)
            if method =='linear_age_step':
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*c_sim - alpha*x1_sim  +add_age_step(age,gamma_l,age_step_loc)
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*c_sim - alpha*x1_sim  +add_age_step_choice(prog_age_choice,gamma_l)
            if method =='linear_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*c_sim - alpha*x1_sim
            if method =='linear_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*c_sim - alpha*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) +add_age_step(age,gamma_l,age_step_loc)
            if method =='linear_age_step_ext':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim +add_age_step(age,gamma_l,age_step_loc)
            if method =='linear_mass_step_ext':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc)
            if method =='linear_mass_age_step_ext':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) +add_age_step(age,gamma_l,age_step_loc)
            if method =='BS20_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc)
            if method =='BS20_age_step':
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc)
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step_choice(prog_age_choice,gamma_l)
            if method =='BS20_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) +add_age_step(age,gamma_l,age_step_loc)
            if method =='BS20_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  #+add_mass_step(np.log10(row['mass']),gamma,step_loc)

            if method =='BS20_fixed_x1_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc)
            if method =='BS20_fixed_x1_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc)
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step_choice(prog_age_choice,gamma_l)
            if method =='BS20_fixed_x1_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) +add_age_step(age,gamma_l,age_step_loc)
            if method =='BS20_fixed_x1_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  #+add_mass_step(np.log10(row['mass']),gamma,step_loc)

            if method =='rv_age_g_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc)
            if method =='rv_age_g_age_step':
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc)
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step_choice(prog_age_choice,gamma_l)
            if method =='rv_age_g_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma_m,mass_step_loc) +add_age_step(age,gamma_l,age_step_loc)
            if method =='rv_age_g_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  #+add_mass_step(np.log10(row['mass']),gamma,step_loc)

            if method =='two_beta_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  #+add_mass_step(np.log10(row['mass']),gamma,step_loc)
            if method =='two_beta_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma,step_loc)
            if method =='two_beta_age_step':
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc) #+add_mass_step(np.log10(row['mass']),gamma,step_loc)
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step_choice(prog_age_choice,gamma_l)

            if method =='two_beta_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc) +add_mass_step(np.log10(row['mass']),gamma,step_loc)

            if method =='two_alpha_no_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  #+add_mass_step(np.log10(row['mass']),gamma,step_loc)
            if method =='two_alpha_mass_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_mass_step(np.log10(row['mass']),gamma,step_loc)
            if method =='two_alpha_age_step':
                #m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc) #+add_mass_step(np.log10(row['mass']),gamma,step_loc)
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim + add_age_step_choice(prog_age_choice,gamma_l)
            if method =='two_alpha_mass_age_step':
                m_obs = M0 + distmod + np.random.normal(0,sigma_int) + beta_sim*cint_sn + (rvs_sn[counter]+1)*Es[counter] - alpha_sim*x1_sim  +add_age_step(age,gamma_l,age_step_loc) +add_mass_step(np.log10(row['mass']),gamma,step_loc)


            self.m_obs.append(m_obs)
            # Fudge some error
            self.mb_err.append(np.max([0.03,np.random.normal(10**(0.4*(m_obs-1.5) - 10)+0.02,0.03)]))

    def set_cint(self,mu,sigma):
        '''

        :param mu:
        :param sigma:
        :return:
        '''
        self.cint= norm(mu,sigma)

    def set_cint_young(self,mu,sigma):
        self.cint_young= norm(mu,sigma)

    def set_cint_old(self,mu,sigma):
        self.cint_old= norm(mu,sigma)
    def choose_rv(self,m,split=10,low=2.75,high=1.5):
        '''

        :param m: stellar mass (in log10 units)
        :type m: float
        :param split: mass split value (in log10 units)
        :type split: float
        :param low: Rv value for low mass galaxies
        :type low: float
        :param high: Rv value for high mass galaxies
        :type high: float
        :return: Rv value for the given mass
        :rtype float

        :example:

        >>> m=9
        >>> split=10
        >>> low=2.8
        >>> choose_rv(m,split)
        2.8

        '''
        return choose_rv(m,split,low,high)

    def setup_sampling(self):
        self.m_obs,self.mb_err,self.x1s,self.cs,self.cints,self.E_SNs,self.masses,self.U_Rs,self.Avs,self.Rvs,self.Rv_SNe,self.zs,self.mean_ages,self.SN_ages,self.betas,self.alphas=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


    ######################
    # Plotting functions #
    ######################
    def plot_galaxy_properties(self):

        f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,4),sharey=True)
        cm =ax1.scatter(self.flux_df['mean_age']/1000,self.flux_df['U_R'],c=self.flux_df['Av'],alpha=0.3)
        ax1.set_xscale('log')
        #ax1.set_xlabel('$\log (M_*/M_{\odot})$',size=20)
        ax1.set_xlabel('Mean Stellar Age (Gyr)',size=20)
        ax1.set_ylabel('U-R',size=20)
        ax1.tick_params(which='both',labelsize=14,right=True,top=True)
        #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])


        cm =ax2.scatter(self.flux_df['ssfr'],self.flux_df['U_R'],c=self.flux_df['Av'],alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_xlabel('$\log (sSFR)$ yr$^{-1}$',size=20)
        #ax2.set_ylabel('U-R',size=20)
        ax2.tick_params(which='both',labelsize=14,right=True,top=True)

        #ax3.set_xscale('log')
        ax3.set_xlabel('$\log (M_*/M_{\odot})$',size=20)

        #ax3.set_ylabel('U-R',size=20)
        ax3.tick_params(which='both',labelsize=14,right=True,top=True)
        #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
        cm =ax3.scatter(np.log10(self.flux_df['mass']),self.flux_df['U_R'],c=self.flux_df['Av'],alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        cb=plt.colorbar(cm,orientation='vertical',ax=ax3,#shrink=0.7
                       )
        cb.set_label('$A_V$',size=20,
                    )

        plt.savefig(aura_dir+'figs/color_vs_host_params')
        # plot U-R v mass only
        f,ax=plt.subplots(figsize=(8,6.5))
        #ax3.set_xscale('log')
        ax.set_xlabel('$\log (M_*/M_{\odot})$',size=20)

        #ax3.set_ylabel('U-R',size=20)
        ax.tick_params(which='both',labelsize=14,right=True,top=True)
        #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
        from matplotlib.colors import ListedColormap
        cm =ax.scatter(np.log10(self.flux_df['mass']),self.flux_df['U_R'],c=self.flux_df['Av'],alpha=0.3,
                       cmap=ListedColormap(sns.color_palette('viridis',n_colors=20).as_hex()))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        cb=plt.colorbar(cm,orientation='horizontal',)#shrink=0.7)
        ax.set_ylabel('U-R',size=20)
        cb.set_label('$A_V$',size=20,
                    )

        plt.tight_layout()
        lisa_colours = pd.read_csv(aura_dir+'5yr-massUR.csv',index_col=0)

        ax.errorbar(lisa_colours['Host Mass'],lisa_colours['Host U-R'],
                    xerr=lisa_colours['Host Mass error'],yerr=lisa_colours['Host U-R error'],
                    linestyle='none',marker='+',label='DES U-R global')

        #ax.scatter(lisa_colours['Host Mass'],lisa_colours['Host U-R']+0.58,
        #
        #           marker='+',label='DES U-R global',c=lisa_colours['Host Mass'],cmap='gist_rainbow')
        ax.legend(loc='upper left',fontsize=15)
        ax.set_ylim(-0.5,3)
        plt.savefig(aura_dir+'figs/U-R_vs_data')


    def plot_x1s(self,df):
        f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,5),sharey=True)
        ax1.scatter(df['mass'],df['x1'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax1.set_xscale('log')
        cm=ax2.scatter(df['U-R'],df['x1'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax3.scatter(df['SN_age'],df['x1'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax3.set_xscale('log')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        cb=plt.colorbar(cm,orientation='vertical',ax=ax3)#shrink=0.7)
        ax1.set_ylabel('$x_1$',size=20)
        cb.set_label('$A_V$',size=20,
                    )
        for ax in [ax1,ax2]:
            ax.tick_params(which='both',direction='in',top=True,right=True)
        ax1.set_xlabel('Stellar Mass',size=20)
        ax2.set_xlabel('$U-R$',size=20)
        ax3.set_xlabel('SN age (Gyr)',size=20)
        ax2.set_xlim(0,2.5)

        f,ax=plt.subplots(figsize=(8,6.5))
        hist=ax.hist(df['x1'],bins=np.linspace(-3,3,100),density=True,label='Simulation',histtype='step',lw=3)
        ax.set_xlabel('$x_1$',size=20)
        ax.hist(MV5yrlowz['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='DES 5yr')
        #ax.hist(pantheon['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='Pantheon')
        ax.legend()
        plt.savefig(aura_dir+'figs/'+'SN_x1_hist_%s'%self.method)
    def plot_cs(self,df):
        f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,5),sharey=True)
        ax1.scatter(df['mass'],df['c'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax1.set_xscale('log')
        cm=ax2.scatter(df['U-R'],df['c'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax3.scatter(df['SN_age'],df['c'],c=df['Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
        ax3.set_xscale('log')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        cb=plt.colorbar(cm,orientation='vertical',ax=ax3)#shrink=0.7)
        ax1.set_ylabel('$c$',size=20)
        cb.set_label('$A_V$',size=20)
        for ax in [ax1,ax2]:
            ax.tick_params(which='both',direction='in',top=True,right=True)
        ax1.set_xlabel('Stellar Mass',size=20)
        ax2.set_xlabel('$U-R$',size=20)
        ax3.set_xlabel('SN age (Gyr)',size=20)
        ax2.set_xlim(0,2)
        ax1.set_ylim(-0.32,0.4)
        f,ax=plt.subplots(figsize=(8,6.5))
        ax.hist(df['c'],bins=np.linspace(-0.3,0.3,25),histtype='step',density=True,label='Sim',lw=3)
        ax.hist(MV5yrlowz['c'],density=True,bins=25,histtype='step',color=split_colour_1,label='Obs DES',lw=3)
        #ax.hist(pantheon['c'],density=True,bins=25,histtype='step',color='y',label='Obs Pantheon',lw=3)
        ax.legend()
        ax.set_xlabel('c',size=20)
        plt.savefig(aura_dir+'figs/'+'SN_c_hist_%s'%self.method)
    def plot_hosts(self,df):
        f,ax=plt.subplots(figsize=(8,6.5))
        ax.scatter(df['mass'],df['Av'],alpha=0.1,c='c',edgecolor='w')
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Mass',size=20)
        ax.set_ylabel('$A_V$',size=20)
        f,ax=plt.subplots(figsize=(8,6.5))
        ax.scatter(df['mass'],df['Rv'],alpha=0.1,c='c',edgecolor='w')
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Mass',size=20)
        ax.set_ylabel('$R_V$',size=20)
        ax.set_ylim(1,6)
        f,ax=plt.subplots(figsize=(8,6.5))
        ax.scatter(df['Av'],df['x1'],alpha=0.1,c='c',edgecolor='w')

        ax.set_ylabel('$x_1$',size=20)
        ax.set_xlabel('$A_V$',size=20)
    def plot_samples(self,zmin=0,zmax=1.2,x1=True,c=True,hosts=True):
        plot_df=self.sim_df[(self.sim_df['z']>zmin)&(self.sim_df['z']<zmax)]
        if x1:
            self.plot_x1s(plot_df)
        if c:
            self.plot_cs(plot_df)
        if hosts:
            self.plot_hosts(plot_df)
