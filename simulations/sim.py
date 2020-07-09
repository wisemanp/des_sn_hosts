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

from des_sn_hosts.utils.utils import compute_HC,compute_features, Constants
from des_sn_hosts.functions import features, match

sns.set_color_codes(palette='colorblind')

class Sim():
    def __init__(self,pop_obj,cat_fn='/media/data3/wiseman/des/photoz/eazy-py/eazy-photoz/outputs/X3_21.eazypy.zout.fits',
        pz_code='eazy',f_deep = 'SN-X3',f_shallow='SN-X2',ccd=21,y=2):
        self.c = Constants()
        self.pop_obj = pop_obj
        if pz_code =='eazy':
            self.small_hostlib = self._get_zphot_res_easy(cat_fn)
        if f_deep:
            self.cat_deep = self._load_deep_cat(f_deep,ccd,y)
            self.gal_pool = self.cat_deep.dropna(subset=['A_IMAGE'])
        if f_shallow:
            self.cat_shallow = self._load_shallow_cat(f_shallow,ccd,y)
        self.root_dir = '/media/data3/wiseman/des/mismatch/'
        self.pop_params = self._get_pop_params()
        self._set_filenames()
    def _set_filenames(self):
        pop_name = self.pop_obj.name
        self.pop_name = pop_name + ''.join(['_%.3e'%v for v in self.pop_params.values()])
        self.pop_fn = self.root_dir+'populations/%s.h5'%self.pop_name
        self.fakes_fn = self.root_dir +'fakes/%s_fakes.h5'%self.pop_name
    def _get_zphot_res_easy(self,cat_fn):
        if cat_fn.split('.')[-1]=='fits':
            zphot_res = Table.read(cat_fn)
            zphot_res.remove_columns(['Avp','massp','SFRp','sSFRp','LIRp'])
            small_hostlib = zphot_res.to_pandas()
            small_hostlib =small_hostlib[(small_hostlib['mass']>0)&(small_hostlib['SFR']>0)&(small_hostlib['mass']<1E16)]
            in_fn = cat_fn.replace('outputs','inputs')
            in_fn = in_fn.replace('eazypy.zout.fits','cat')
            zphot_in = Table.read(in_fn,format='ascii').to_pandas()
            small_hostlib = small_hostlib.merge(zphot_in,on='id',how='inner')
        elif cat_fn.split('.')[-1]=='csv':
            small_hostlib= pd.read_csv(cat_fn)
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
        cat_deep.rename(index=str,columns={
                                    'X_WORLD':'RA','Y_WORLD':'DEC',
                                   'z':'SPECZ',
                                   'ez':'SPECZ_ERR',
                                   'source':'SPECZ_CATALOG',
                                   'flag':'SPECZ_FLAG',
                                   'objtype_ozdes':'OBJTYPE_OZDES',
                                   'transtype_ozdes':'TRANSTYPE_OZDES',
                                   'MAG_APER_g':'MAG_APER_4_G',
                                   'MAG_APER_r':'MAG_APER_4_R',
                                   'MAG_APER_i':'MAG_APER_4_I',
                                   'MAG_APER_z':'MAG_APER_4_Z',
                                   'MAGERR_APER_g':'MAGERR_APER_4_G',
                                   'MAGERR_APER_r':'MAGERR_APER_4_R',
                                   'MAGERR_APER_i':'MAGERR_APER_4_I',
                                   'MAGERR_APER_z':'MAGERR_APER_4_Z',
                                   'MAGERR_SYST_APER_g':'MAGERR_SYST_APER_4_G',
                                   'MAGERR_SYST_APER_r':'MAGERR_SYST_APER_4_R',
                                   'MAGERR_SYST_APER_i':'MAGERR_SYST_APER_4_I',
                                   'MAGERR_SYST_APER_z':'MAGERR_SYST_APER_4_Z',
                                   'MAGERR_STATSYST_APER_g':'MAGERR_STATSYST_APER_4_G',
                                   'MAGERR_STATSYST_APER_r':'MAGERR_STATSYST_APER_4_R',
                                   'MAGERR_STATSYST_APER_i':'MAGERR_STATSYST_APER_4_I',
                                   'MAGERR_STATSYST_APER_z':'MAGERR_STATSYST_APER_4_Z',
                                   'MAG_AUTO_g':'MAG_AUTO_G',
                                   'MAG_AUTO_r':'MAG_AUTO_R',
                                   'MAG_AUTO_i':'MAG_AUTO_I',
                                   'MAG_AUTO_z':'MAG_AUTO_Z',
                                   'MAGERR_AUTO_g':'MAGERR_AUTO_G',
                                   'MAGERR_AUTO_r':'MAGERR_AUTO_R',
                                   'MAGERR_AUTO_i':'MAGERR_AUTO_I',
                                   'MAGERR_AUTO_z':'MAGERR_AUTO_Z',
                                   'MAGERR_SYST_AUTO_g':'MAGERR_SYST_AUTO_G',
                                   'MAGERR_SYST_AUTO_r':'MAGERR_SYST_AUTO_R',
                                   'MAGERR_SYST_AUTO_i':'MAGERR_SYST_AUTO_I',
                                   'MAGERR_SYST_AUTO_z':'MAGERR_SYST_AUTO_Z',
                                   'MAGERR_STATSYST_AUTO_g':'MAGERR_STATSYST_AUTO_G',
                                   'MAGERR_STATSYST_AUTO_r':'MAGERR_STATSYST_AUTO_R',
                                   'MAGERR_STATSYST_AUTO_i':'MAGERR_STATSYST_AUTO_I',
                                   'MAGERR_SSTATYST_AUTO_z':'MAGERR_STATSYST_AUTO_Z',
                                   'FLUX_AUTO_g':'FLUX_AUTO_G',
                                    'FLUX_AUTO_r':'FLUX_AUTO_R',
                                    'FLUX_AUTO_i':'FLUX_AUTO_I',
                                    'FLUX_AUTO_z':'FLUX_AUTO_Z',
                                   'FLUXERR_AUTO_g':'FLUXERR_AUTO_G',
                                    'FLUXERR_AUTO_r':'FLUXERR_AUTO_R',
                                    'FLUXERR_AUTO_i':'FLUXERR_AUTO_I',
                                    'FLUXERR_AUTO_z':'FLUXERR_AUTO_Z',
                                   'FLUX_APER_g':'FLUX_APER_4_G',
                                    'FLUX_APER_r':'FLUX_APER_4_R',
                                    'FLUX_APER_i':'FLUX_APER_4_I',
                                    'FLUX_APER_z':'FLUX_APER_4_Z',
                                    'FLUXERR_APER_g':'FLUXERR_APER_4_G',
                                    'FLUXERR_APER_r':'FLUXERR_APER_4_R',
                                    'FLUXERR_APER_i':'FLUXERR_APER_4_I',
                                    'FLUXERR_APER_z':'FLUXERR_APER_4_Z',
                                   'CLASS_STAR_g':'CLASS_STAR_G',
                                   'CLASS_STAR_r':'CLASS_STAR_R',
                                   'CLASS_STAR_i':'CLASS_STAR_I',
                                   'CLASS_STAR_z':'CLASS_STAR_Z',
                                   'MAG_ZEROPOINT_g':'MAG_ZEROPOINT_G',
                                   'MAG_ZEROPOINT_r':'MAG_ZEROPOINT_R',
                                   'MAG_ZEROPOINT_i':'MAG_ZEROPOINT_I',
                                   'MAG_ZEROPOINT_z':'MAG_ZEROPOINT_Z',
                                   'LIMMAG_g':'LIMMAG_G',
                                   'LIMMAG_r':'LIMMAG_R',
                                   'LIMMAG_i':'LIMMAG_I',
                                   'LIMMAG_z':'LIMMAG_Z',
                                   'LIMFLUX_g':'LIMFLUX_G',
                                   'LIMFLUX_r':'LIMFLUX_R',
                                   'LIMFLUX_i':'LIMFLUX_I',
                                   'LIMFLUX_z':'LIMFLUX_Z',
                                   'MAG_ZEROPOINT_ERR_g':'MAG_ZEROPOINT_ERR_G',
                                   'MAG_ZEROPOINT_ERR_r':'MAG_ZEROPOINT_ERR_R',
                                   'MAG_ZEROPOINT_ERR_i':'MAG_ZEROPOINT_ERR_I',
                                   'MAG_ZEROPOINT_ERR_z':'MAG_ZEROPOINT_ERR_Z'
        },inplace=True)

        cat_deep.drop([
                'FLUX_RADIUS_g','FLUX_RADIUS_r','FLUX_RADIUS_i','FLUX_RADIUS_z',
               'FWHM_WORLD_g','FWHM_WORLD_r','FWHM_WORLD_i','FWHM_WORLD_z'],axis=1,inplace=True)
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
        if not os.path.isdir(self.root_dir+'populations'):
            os.mkdir(self.root_dir+'populations')
        self.pop_df.to_hdf(self.pop_fn,key='default_pop')
        print('Saved population DataFrame to %s'%self.pop_fn)
        return pop
    def load_pop(self,fn=None,key='default_pop'):
        if fn==None:
            fn =self.root_dir+'populations/%s.h5'%self.pop_name
        self.pop_df = pd.read_hdf(fn,key=key)
    def plot_pop(self,ax=None):
        if not ax:
            f,ax=plt.subplots(figsize=(12,7))
        ax.scatter(self.pop_obj.distances,self.pop_obj.latent_fluxes,alpha=0.05,label='Popsynth')
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
        plt.savefig(self.root_dir+'figs/%s_drawn_population'%self.pop_name) #add population name!!!

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

        fakes.to_hdf(self.fakes_fn,key='fakes')
        self.fakes = fakes

        return fakes

    def match_fakes(self):
        print('Matching fakes to hosts...')
        matched_fn = match.main(fn = self.fakes_fn,resdir = self.root_dir+'fakes/%s/'%self.pop_name)
        print('Successfully matched fakes to hosts. Reading in the match!')
        self.matched_fakes = pd.read_csv(matched_fn,names=[
            'ANGSEP',
            'A_IMAGE',
            'B_IMAGE',
            'CCDNUM',
            'CLASS_STAR_g',
            'CLASS_STAR_i',
            'CLASS_STAR_r',
            'CLASS_STAR_z',
            'CXX_IMAGE',
            'CXY_IMAGE',
            'CYY_IMAGE',
            'DLR',
            'DLR_RANK',
            'EDGE_FLAG',
            'ELONGATION',
            'FIELD',
            'FLUXERR_APER_g',
            'FLUXERR_APER_i',
            'FLUXERR_APER_r',
            'FLUXERR_APER_z',
            'FLUXERR_AUTO_g',
            'FLUXERR_AUTO_i',
            'FLUXERR_AUTO_r',
            'FLUXERR_AUTO_z',
            'FLUX_APER_g',
            'FLUX_APER_i',
            'FLUX_APER_r',
            'FLUX_APER_z',
            'FLUX_AUTO_g',
            'FLUX_AUTO_i',
            'FLUX_AUTO_r',
            'FLUX_AUTO_z',
            'FLUX_RADIUS_g',
            'FLUX_RADIUS_i',
            'FLUX_RADIUS_r',
            'FLUX_RADIUS_z',
            'FWHM_WORLD_g',
            'FWHM_WORLD_i',
            'FWHM_WORLD_r',
            'FWHM_WORLD_z',
            'GALID_true',
            'KRON_RADIUS',
            'LIMFLUX_g',
            'LIMFLUX_i',
            'LIMFLUX_r',
            'LIMFLUX_z',
            'LIMMAG_g',
            'LIMMAG_i',
            'LIMMAG_r',
            'LIMMAG_z',
            'MAGERR_APER_g',
            'MAGERR_APER_i',
            'MAGERR_APER_r',
            'MAGERR_APER_z',
            'MAGERR_AUTO_g',
            'MAGERR_AUTO_i',
            'MAGERR_AUTO_r',
            'MAGERR_AUTO_z',
            'MAGERR_STATSYST_APER_g',
            'MAGERR_STATSYST_APER_i',
            'MAGERR_STATSYST_APER_r',
            'MAGERR_STATSYST_APER_z',
            'MAGERR_STATSYST_AUTO_g',
            'MAGERR_STATSYST_AUTO_i',
            'MAGERR_STATSYST_AUTO_r',
            'MAGERR_STATSYST_AUTO_z',
            'MAGERR_SYST_APER_g',
            'MAGERR_SYST_APER_i',
            'MAGERR_SYST_APER_r',
            'MAGERR_SYST_APER_z',
            'MAGERR_SYST_AUTO_g',
            'MAGERR_SYST_AUTO_i',
            'MAGERR_SYST_AUTO_r',
            'MAGERR_SYST_AUTO_z',
            'MAG_APER_g',
            'MAG_APER_i',
            'MAG_APER_r',
            'MAG_APER_z',
            'MAG_AUTO_g',
            'MAG_AUTO_i',
            'MAG_AUTO_r',
            'MAG_AUTO_z',
            'MAG_ZEROPOINT_ERR_g',
            'MAG_ZEROPOINT_ERR_i',
            'MAG_ZEROPOINT_ERR_r',
            'MAG_ZEROPOINT_ERR_z',
            'MAG_ZEROPOINT_g',
            'MAG_ZEROPOINT_i',
            'MAG_ZEROPOINT_r',
            'MAG_ZEROPOINT_z',
            'MY',
            'PHOTOZ',
            'PHOTOZ_ERR',
            'SNID',
            'THETA_IMAGE',
            'X_IMAGE',
            'X_WORLD',
            'Y_IMAGE',
            'Y_WORLD',
            'Z_RANK',
            'ez',
            'flag',
            'objtype_ozdes',
            'source',
            'transtype_ozdes',
            'z',
            'SN_RA',
            'SN_DEC'],
            index_col =0)
        self.matched_fakes.reset_index(drop=False,inplace=True)
        self.matched_fakes.rename(columns={'index':'GALID_obs'},inplace=True)
        self.matched_fakes.replace(np.NaN,-9999,inplace=True)
        self.matched_fakes.drop_duplicates(subset=['GALID_true','GALID_obs','SNID','Z_RANK'],inplace=True)
        self.matched_fn = matched_fn.replace('result','h5')
        print('Writing the matched fakes to file: %s'%self.matched_fn)
        self.matched_fakes.to_hdf(self.matched_fn,key='fakes')
        self.matched_fn, self.matched_fakes_features = features.main(fn=self.matched_fn,key='fakes')

    def prep_rf(self):

        DLRs = self.fakes['DLR']
        self.matched_fakes = self.matched_fakes_features.merge(self.small_hostlib.rename(columns={'id':'GALID_obs'}),on='GALID_obs',how='left')
        self.matched_fakes_features['GALID_diff'] = self.matched_fakes_features['GALID_true'] -self.matched_fakes_features['GALID_obs']
        self.matched_fakes_features = self.matched_fakes_features.merge(self.fakes.drop(['GAL_RA','GAL_DEC','GALID'],axis=1).rename(columns={'ID':'SNID','DLR':'DLR_true'}),on='SNID',how='inner',suffixes=['_obs','_true'])
        self.matched_closest = self.matched_fakes_features[((self.matched_fakes_features['DLR_RANK']==1)|(self.matched_fakes_features['DLR_RANK']==-1))&(self.matched_fakes_features['Z_RANK']<2)]
        self.matched_closest =self.matched_closest[(self.matched_closest['HC']<95)&(self.matched_closest['HC']>-95)]
        self.matched_closest.drop_duplicates('SNID',keep='last',inplace=True)
        self.matched_hosts = self.matched_closest[(self.matched_closest['DLR_RANK']==1)&(self.matched_closest['Z_RANK']<2)]
        self.matched_hostless = self.matched_closest[(self.matched_closest['DLR_RANK']==-1)&(self.matched_closest['Z_RANK']<2)&(self.matched_closest['GALID_true']>=0)]
        self.matched_truehostless = self.matched_closest[self.matched_closest['GALID_true']<0]
    def train_rf(self,sf,cv=True,**kwargs):
        self.classifier = Classifier(self.matched_closest,self.root_dir+'config/config_classifier.yaml')
        if cv:
            self.classifier.CV(sf=sf,**kwargs)
        else:
            self.classifier.load_clf(sf)

        self.classifier.fit_test()


class ZPowerCosmoSchechterSim(Sim):

    def __init__(self,Lstar,alpha,Lambda,delta=0,r_max=2,cat_fn='/media/data3/wiseman/des/photoz/eazy-py/eazy-photoz/outputs/X3_21.eazypy.zout.fits',
        pz_code='eazy',f_deep = 'SN-X3',f_shallow='SN-X2',ccd=21,y=2,):

        pop_obj = popsynth.populations.SchechterZPowerCosmoPopulation(Lmin=Lstar, alpha=alpha,Lambda=Lambda,r_max=r_max,delta=delta)

        super(ZPowerCosmoSchechterSim, self).__init__(
            pop_obj,
            cat_fn=cat_fn,
            pz_code=pz_code,
            f_deep=f_deep,
            f_shallow=f_shallow,
            ccd=ccd,
            y=y
        )
