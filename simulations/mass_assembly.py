import numpy as np
import pandas as pd
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo
import astropy.io.fits as fits
import os
import scipy.stats as stats
from scipy.special import erf
from shutil import copyfile
from astropy import units as u
from astropy import wcs
import aplpy
import numpy as np
import subprocess
import glob
import matplotlib.pyplot as plt
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
import argparse
plt.rcParams['errorbar.capsize']=4
sns.set_context('paper')
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': [16,9]})
plt.rcParams.update({'xtick.direction':'in'})
plt.rcParams.update({'ytick.direction':'in'})
#plt.rcParams.update({'axes.locator'})

def phi_t(t,tp,alpha,s):
    '''Functional form of the delay time distribution'''
    return (t/tp)**alpha / ((t/tp)**(alpha-s)+1)


def psi_Mz(M,z):
    #print("Nominal SFRs: \n",2.00*np.exp(1.33*z)*(M/1E+10)**0.7)
    return 2.00*np.exp(1.33*z)*(M/1E+10)**0.7

def logMQ_z(z):
    Mlo,zlo = 10.43,0.9
    Mhi,zhi = 8.56,5.6
    #print ("Quenching masses: \n",((Mlo+zlo*np.log10(1+z))*(z<=1.5)) + ((Mhi+zhi*np.log10(1+z))*(z>1.5)))
    return ((Mlo+zlo*np.log10(1+z))*(z<=1.5)) + ((Mhi+zhi*np.log10(1+z))*(z>1.5))

def pQ_Mz(M,z):
    #print("Quenching function: \n",0.5*(1-erf((np.log10(M)-logMQ_z(z))/1.5)))
    return 0.5*(1-erf((np.log10(M)-logMQ_z(z))/1.5))

def sfr_Mz(M,z):
    return pQ_Mz(M,z) * psi_Mz(M,z)


def psi_Mz_alt(M,z):
    #print("Alternate SFRs: \n",36.4*(M/1E+10)**0.7 * np.exp(1.9*z)/(np.exp(1.7*z)+np.exp(0.2*z)))
    return 36.4*(M/1E+10)**0.7 * np.exp(1.9*z)/(np.exp(1.7*z)+np.exp(0.2*z))

def logMQ_z_alt(z):

    #print ("Quenching masses: \n",10.077 + 0.636*z)
    return 10.077 + 0.636*z

def pQ_Mz_alt(M,z):
    #print("Quenching function: \n",0.5*(1-erf((np.log10(M)-logMQ_z_alt(z))/1.5)))
    return 0.5*(1-erf((np.log10(M)-logMQ_z_alt(z))/1.5))

def pmin_z(z):
    return 1-((z-10)/10)**2

def pQ_Mz_ft(M,z):
    return pmin_z(z) + (1 - pmin_z(z))*pQ_Mz_alt(M,z)

def fml_t(t):
    return 0.046*np.log((t/0.276)+1)

def sfr_Mz_alt(M,z):
    return pQ_Mz_ft(M,z) * psi_Mz_alt(M,z)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt','--dt',help='Time step (Myr)',default=0.5,dtype=float)
    parser.add_argument('-es','--early_step',help='Early Universe T_F step (Myr)',default=25,dtype=float)
    parser.add_argument('-ls','--late_step',help='Late Universe T_F step (Myr)',default=50,dtype=float)
    args = parser.parse_args()
    return args

def main(args):
    config=yaml.load(open(args.config))
    save_dir = config['rates_root']+'SFHs/'
    dt = args.dt
    tfs = np.concatenate([np.arange(1000,10000,args.late_step),np.arange(10000,13000,args.early_step)])

    for tf in tqdm(tfs):
        print("Starting epoch: %.f Myr "%tf)

        ages = np.arange(0,tf+dt,dt)
        m = 1E+6
        m_formed = []
        m_lost_tot = [0]#0.
        m_arr = []
        ts = []
        for counter,age in enumerate(ages):
            t = tf-age
            ts.append(t)
            #print("Current epoch: %.f Myr"%t)
            try:
                z_t = z_at_value(cosmo.lookback_time,t*u.Myr,zmin=0)
            except:
                z_t = 0
            #print("current redshift: %.2f"%z_t)
            m_created = sfr_Mz(m,z_t)*dt*1E+6
            m_formed.append(m_created)

            #print("Mass formed in the last %3f Myr: %2g Msun"%(dt,m_created))
            #print("Mass formed at each epoch so far: ",m_formed)
            taus = ages[:counter+1][::-1]
            #print("Time since epochs of star formation: ",taus)
            f_ml= fml_t(taus)
            #print("Fractional mass lost since each epoch",f_ml)
            ml = f_ml * m_formed
            #m_lost_tot = np.concatenate([m_lost_tot,[0]])

            if counter>1:
                #print('trying to subtract m_lost_tot',m_lost_tot,'from ml ',ml)
                new_ml = np.sum(ml[:counter]- m_lost_tot)
            else:
                new_ml = np.sum(ml)
            #print("New mass loss this cycle",new_ml)
            m_lost_tot = ml
            #print("Current array of masses lost",[ "{:0.2e}".format(x) for x in m_lost_tot ])
            #ml_tot = ml - m_lost
            #m_lost = ml_tot
            m = m + m_created - new_ml
            #print("Final mass of this epoch: %.1e"%m)
            #print("#"*100)
            m_arr.append(m)
        #print (m_formed)
        #print(m_lost_tot)
        final_age_weights = m_formed - m_lost_tot
        track = np.array([ts,ages,m_formed,final_age_weights,m_arr]).T
        pd.DataFrame(track,columns=['t','age','m_formed','final_age_weights','m_tot']).to_hdf(os.path.join(save_dir,'SFHs_%.1f.h5'%dt,key='%3.0f'%tf))

if __name__=="__main__":
    main(parser())    
