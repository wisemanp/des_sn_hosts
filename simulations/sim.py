import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import time
import itertools
import progressbar
import os
import pickle
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import popsynth
import networkx as nx
import warnings
warnings.simplefilter('ignore')

from des_mismatch.utils.utils import compute_HC,compute_features, Constants


sns.set_color_codes(palette='colorblind')

class Sim():
    def __init__(self,pop_obj,cat_fn='/media/data3/wiseman/des/photoz/eazy-py/eazy-photoz/outputs/X3_21.eazypy.zout.fits',
        pz_code='eazy',f_deep = 'SN-X3',f_shallow='SN-X2',ccd=21,y=2):
        self.c = Constants()
        self.pop_obj = pop_obj
        if pz_code =='eazy':
            self.small_hostlib = self._get_zphot_res_easy(cat_fn)
        self.cat_deep = self._load_deep_cat(f_deep,ccd,y)
        self.gal_pool = self.cat_deep.dropna(subset=['A_IMAGE'])
        self.cat_shallow = self._load_shallow_cat(f_shallow,ccd,y)
        self.root_dir = '/media/data3/wiseman/des/mismatch/'
        self.pop_params = self._get_pop_params()

    def _get_zphot_res_easy(self,cat_fn):
        zphot_res = Table.read(cat_fn)
        zphot_res.remove_columns(['Avp','massp','SFRp','sSFRp','LIRp'])
        small_hostlib = zphot_res.to_pandas()
        small_hostlib =small_hostlib[(small_hostlib['mass']>0)&(small_hostlib['SFR']>0)&(small_hostlib['mass']<1E16)]
        in_fn = cat_fn.replace('outputs','inputs')
        in_fn = in_fn.replace('eazypy.zout.fits','cat')
        zphot_in = Table.read(in_fn,format='ascii').to_pandas()
        small_hostlib = small_hostlib.merge(zphot_in,on='id',how='inner')
        return small_hostlib
    def _get_zphot_res_zpeg(self,cat_fn):
        pass

    def _load_deep_cat(self,f,ccd,y):
        cat_deep = pd.read_csv(os.path.join('/media/data3/wiseman/des/coadding/5yr_stacks/MY%s'%y,f,'CAP',str(ccd),
            '%s_%s_%s_obj_deep_v7.cat'%(y,f,ccd)),index_col=0)
        cat_deep.replace(-9999.000000,np.NaN,inplace=True)
        cat_deep.dropna(subset=['MAGERR_AUTO_r'],inplace=True)
        cat_deep.reset_index(drop=False,inplace=True)
        cat_deep = cat_deep[(cat_deep['X_IMAGE']>200)&(cat_deep['X_IMAGE']<4200)&(cat_deep['Y_IMAGE']>80)&(cat_deep['Y_IMAGE']<2080)]
        cat_deep.rename(columns={'index':'id'},inplace=True)
        cat_deep = cat_deep.merge(self.small_hostlib,on='id',how='outer')
        return cat_deep

    def _load_shallow_cat(self,f,ccd,y):
        cat_shallow = pd.read_csv(os.path.join('/media/data3/wiseman/des/coadding/5yr_stacks/MY%s'%y,f,'CAP',str(ccd),
            '%s_%s_%s_obj_deep_v7.cat'%(y,f,ccd)),index_col=0)
        cat_shallow.replace(-9999.000000,np.NaN,inplace=True)
        cat_shallow.dropna(subset=['MAGERR_AUTO_r'],inplace=True)
        cat_shallow = cat_shallow[(cat_shallow['X_IMAGE']>150)&(cat_shallow['X_IMAGE']<4200)&(cat_shallow['Y_IMAGE']>50)&(cat_shallow['Y_IMAGE']<2080)]
        return cat_shallow

    def _load_hostlibs(self):
        # Load in HOSTLIBs so we can estiamte the DLR distribution
        self.sv_hostlib =pd.read_csv(self.root_dir+'MV_SVAvsMICECAT/1_SIM/DESSVA/PIP_MV_DET_EFF_DESSVA/PIP_MV_DET_EFF_DESSVA.DUMP',
                                comment='#',skiprows=0,delimiter='\s+')
        self.micecat_hostlib =pd.read_csv(self.root_dir+'MV_SVAvsMICECAT/1_SIM/MICE5/PIP_MV_DET_EFF_MICE5/PIP_MV_DET_EFF_MICE5.DUMP',
                                     comment='#',skiprows=0,delimiter='\s+')

    def _get_pop_params(self):
        params = self.pop_obj._params
        for k,v in params.items():
            if type(v) ==u.quantity.Quantity:
                params[k] = v.value
        return params
    def synth_pop(self):
        pop = self.pop_obj.draw_survey(boundary = self.c.fluxlim_ergcms_des,hard_cut=True,flux_sigma=0.1)
        self.pop_df = pd.DataFrame(np.array([pop.distances,pop.luminosities/self.c.Lsun,pop.latent_fluxes]).T,columns=['z','Lv','Fv'])
        # save the file
        if not os.path.isdir(root_dir+'populations'):
            os.path.mkdir(root_dir+'populations')
        pop_name = self.pop_df.name

        pop_name.extend(['_%s'%v for v in self.pop_params.values()])
        self.pop_df.to_csv(root_dir+'populations/%s'pop_name)
    def plot_pop(self,ax=None):
        if not ax:
            f,ax1=plt.subplots(figsize=(12,7))
        ax.scatter(population.distances,population.latent_fluxes,alpha=0.05,label='Popsynth')
        ax.set_ylim(1E-22,1E-10)
        ax.set_yscale('log')

        flux_ergcmsa = 2.99792458E-11*self.small_hostlib['F296']/(self.c.des_filters['i'][1]**2)
        flux_ergcms = flux_ergcmsa*(self.c.des_filters['i'][2]-self.c.des_filters['i'][0])
        ax.scatter(self.small_hostlib['z_phot'],flux_ergcms,alpha=0.1,color='b',marker='^',label='Data + photo_Z')
        ax.hlines(fluxlim_ergcms,0,2,linestyle='--',lw=2,color='w',label='DES-SN flux limit')
        leg=ax1.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        ax.set_xlabel('Redshift',size=18)
        ax.set_ylabel('$i$ band flux (erg/cm/s)',size=18)
        plt.savefig(self.root_dir+'drawn_population_test') #add population name!!!

    def gen_fakes(self,n_samples=1E+5):
        RAs,DECs,GAL_RAs,GAL_DECs,DLRs,GALID,Pz,mass,ID =[],[],[],[],[],[],[],[],[]
        self.gal_pool['logLv'] = np.log10(self.gal_pool['Lv'])
        lvs = []
        selected_galaxy_inds = []
        n_samples= int(n_samples)
        sample_df = self.pop_df.sample(n_samples)
        minra,maxra,mindec,maxdec = self.cat_deep.RA.min(),self.cat_deep.RA.max(),self.cat_deep.DEC.min(),self.cat_deep.DEC.max()
        with progressbar.ProgressBar(max_value=n_samples) as bar:
            for counter,i in enumerate(sample_df.index):
                simgal = sample_df.loc[i]
                lv =simgal['Lv']
                z = simgal['z']
                fv = simgal['Fv']
                lvs.append(lv)
                loglv = np.log10(lv)
                try:
                    z_lum_match = gal_pool[(gal_pool['logLv']>loglv-0.1)&(gal_pool['logLv']<loglv+0.1)&\
                                           (gal_pool['z_phot']<z+0.05)&(gal_pool['z_phot']>z-0.05)]
                    galaxy = z_lum_match.iloc[np.random.randint(len(z_lum_match.index))]
                    d_DLR = stats.lognorm(.62,loc=0.0,scale=1).rvs()
                    DLRs.append(d_DLR)
                    RA,DEC = generate_sn_loc(galaxy,d_DLR)
                    RAs.append(RA)
                    DECs.append(DEC)
                    GALID.append(galaxy['id'])
                    GAL_RAs.append(galaxy['RA'])
                    GAL_DECs.append(galaxy['DEC'])
                    Pz.append(galaxy['z_phot'])
                    ID.append(i)
                    mass.append(galaxy['mass'])
                except:
                    RA = (maxra - minra) * np.random.random_sample() + minra
                    DEC = (maxdec - mindec) * np.random.random_sample() + mindec
                    RAs.append(RA)
                    DECs.append(DEC)
                    DLRs.append(0)
                    GALID.append(-1*i)
                    GAL_RAs.append(-9999)
                    GAL_DECs.append(-9999)
                    Pz.append(z)
                    ID.append(counter)
                    mass.append(3*lv) # M/L = 3
                bar.update(counter)

        fakes = pd.DataFrame(columns=['ID','GALID','SN_RA','SN_DEC','GAL_RA','GAL_DEC','DLR','PZ','mass'])
        fakes['SN_RA'] = RAs
        fakes['SN_DEC'] = DECs
        fakes['GAL_RA'] = GAL_RAs
        fakes['GAL_DEC'] = GAL_DECs
        fakes['ID'] = ID
        fakes['GALID'] = GALID
        fakes['DLR'] = DLRs
        fakes['PZ'] = Pz
        fakes['mass'] = mass
        fakes.to_csv('/media/data3/wiseman/des/mismatch/X3_21_fakes_with_faint.csv')
        self.fakes = fakes
        return fakes

class ZPowerSchechterSim(Sim):

    def __init__(self,Lstar,alpha,Lambda,delta=0,r_max=2,cat_fn='/media/data3/wiseman/des/photoz/eazy-py/eazy-photoz/outputs/X3_21.eazypy.zout.fits',
        pz_code='eazy',f_deep = 'SN-X3',f_shallow='SN-X2',ccd=21,y=2,):

        pop_obj = popsynth.populations.SchechterZPowerCosmoPopulation(Lmin=Lstar, alpha=alpha,Lambda=Lambda,r_max=r_max,delta=delta)

        super(ZPowerSchechterSim, self).__init__(
            pop_obj,
            cat_fn=cat_fn,
            pz_code=pz_code,
            f_deep=f_deep,
            f_shallow=f_shallow,
            ccd=ccd,
            y=y
        )
