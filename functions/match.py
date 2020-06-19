# -*- coding: utf-8 -*-

import multiprocessing
from multiprocessing import Process
import os
import subprocess
import time
import numpy as np
import logging
import progressbar
import tqdm
import pandas as pd
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import itertools
import warnings
warnings.simplefilter('ignore')

hashes = '#'*100

def match_fakes(galid,snid,ra,dec,resdir,dist_thresh = 5,y=2,chip=21,f='SN-X3'):
    logger = logging.getLogger(__name__)
    logger.handlers =[]
    ch = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    formatter =logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    #logger.info(hashes)
    #logger.info("Entered 'cap_sn_lookup' to do find host galaxy candidates for %s"%snid)
    #logger.info(hashes)
    bands = ['g','r','i','z']


    my = 'MY'+str(y)
    main_res_df = pd.DataFrame()
    logger.debug('Looking in chips %s, %s, %s'%(chip -1, chip,chip+1))
    add_lim=False
    for ch in [chip -1, chip,chip+1]:
        ind=None
        capres_fn = os.path.join('/media/data3/wiseman/des/coadding/5yr_stacks',my,
                             f,'CAP',str(ch),'%s_%s_%s_obj_deep_v7.cat'%(y,f,ch))
        capres = pd.read_csv(capres_fn,index_col = 0)

        add_lim=False

        if len(capres)==0:
            logger.debug('The capres  %s has no length'%capres_fn)
        logger.debug('Managed to read in the catalog %s'%capres_fn)
        search_rad = dist_thresh
        capres_box = capres[(capres['X_WORLD']< ra+(search_rad/3600))&(capres['X_WORLD']> ra-(search_rad/3600)) & (capres['Y_WORLD']> dec-(search_rad/3600)) & (capres['Y_WORLD']< dec+(search_rad/3600))]
        logger.debug('Found %s galaxies within a search box %.2f arsecs wide'%(len(capres_box.index.unique()),search_rad*2))
        cols = capres_box.columns.tolist() + [
            'SNID',
            'GALID_true',
             'DLR',
             'DLR_RANK',
             'ANGSEP',
             'EDGE_FLAG'
        ]
        res_df = pd.DataFrame(columns=cols)
        res_df['EDGE_FLAG'] = 0
        sncoord = SkyCoord(ra = ra*u.deg,dec = dec*u.deg)
        catalog = SkyCoord(ra = capres_box.X_WORLD.values*u.deg,dec = capres_box.Y_WORLD.values*u.deg)
        d2d= sncoord.separation(catalog)
        close_inds = d2d <dist_thresh*u.arcsec
        dists = d2d[close_inds]
        match = capres_box.iloc[close_inds]
        angsep = np.array([float(d2d[close_inds][j].to_string(unit=u.arcsec,decimal=True)) for j in range(len(d2d[close_inds]))])
        hashost = 0
        lims = True
        limcols = ['X_WORLD', 'Y_WORLD', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO_g',
           'MAGERR_AUTO_g', 'MAG_APER_g', 'MAGERR_APER_g', 'FLUX_AUTO_g',
           'FLUXERR_AUTO_g', 'FLUX_APER_g', 'FLUXERR_APER_g', 'FWHM_WORLD_g',
           'ELONGATION', 'KRON_RADIUS', 'CLASS_STAR_g', 'FLUX_RADIUS_g', 'A_IMAGE',
           'B_IMAGE', 'THETA_IMAGE', 'CXX_IMAGE', 'CYY_IMAGE', 'CXY_IMAGE',
           'MAGERR_SYST_AUTO_g', 'MAGERR_SYST_APER_g', 'MAGERR_STATSYST_AUTO_g',
           'MAGERR_STATSYST_APER_g',
           'PHOTOZ', 'PHOTOZ_ERR', 'MAG_AUTO_r',
           'MAGERR_AUTO_r', 'MAG_APER_r', 'MAGERR_APER_r', 'FLUX_AUTO_r',
           'FLUXERR_AUTO_r', 'FLUX_APER_r', 'FLUXERR_APER_r', 'FWHM_WORLD_r',
           'CLASS_STAR_r', 'FLUX_RADIUS_r', 'MAGERR_SYST_AUTO_r',
           'MAGERR_SYST_APER_r', 'MAGERR_STATSYST_AUTO_r',
           'MAGERR_STATSYST_APER_r',
           'MAG_AUTO_i', 'MAGERR_AUTO_i', 'MAG_APER_i', 'MAGERR_APER_i',
           'FLUX_AUTO_i', 'FLUXERR_AUTO_i', 'FLUX_APER_i', 'FLUXERR_APER_i',
           'FWHM_WORLD_i', 'CLASS_STAR_i', 'FLUX_RADIUS_i', 'MAGERR_SYST_AUTO_i',
           'MAGERR_SYST_APER_i', 'MAGERR_STATSYST_AUTO_i',
           'MAGERR_STATSYST_APER_i',
           'MAG_AUTO_z', 'MAGERR_AUTO_z', 'MAG_APER_z', 'MAGERR_APER_z',
           'FLUX_AUTO_z', 'FLUXERR_AUTO_z', 'FLUX_APER_z', 'FLUXERR_APER_z',
           'FWHM_WORLD_z', 'CLASS_STAR_z', 'FLUX_RADIUS_z', 'MAGERR_SYST_AUTO_z',
           'MAGERR_SYST_APER_z', 'MAGERR_STATSYST_AUTO_z',
           'MAGERR_STATSYST_APER_z','DLR', 'DLR_RANK',
           'ANGSEP','z','ez','flag','source','objtype_ozdes','transtype_ozdes','Z_RANK']
        if len(match)>0:
            logger.debug('Found a host!')
            lims = False
            res_df = res_df.append(match)
            res_df['GALID_true']=galid
            res_df['SNID']=snid
            dlr = get_DLR_ABT(ra,dec, match.X_WORLD, match.Y_WORLD, match['A_IMAGE'], match['B_IMAGE'],  match['THETA_IMAGE'], angsep)[0]

            res_df['ANGSEP'] = angsep

            res_df['DLR'] = np.array(dlr)
            rank = res_df['DLR'].rank(method='dense').astype(int)

            for counter, r in enumerate(res_df['DLR'].values):
                if r >4:
                    rank.iloc[counter]*=-1
            res_df['DLR_RANK']=rank
            if len(match)>5:
                res_df = res_df[res_df['DLR']<30]

        if len(match)==0 or len(res_df)==0:
            lims=True
            logger.debug('Didnt find a host! Reporting limits')
            if ch ==chip:
                res_df = res_df.append(capres.iloc[0])

                res_df[limcols] = np.NaN
                res_df.SNID = snid
                res_df.GALID_true = galid
                res_df.name=0

        if lims:
            ind = res_df.index

        else:
            ind = res_df[res_df['DLR_RANK']==1].index

        if len(res_df[res_df['DLR_RANK']==1])==0 and ch == chip:
            try:
                add_lim =True
                logger.debug('Adding lim in chip %s'%ch)
                ind = [res_df.index.max()+1]
                logger.debug('Adding a single row with ind %s'%ind)
                lim_row = res_df.iloc[0]
                lim_row.name=ind[0]
                lim_row[limcols] = np.NaN
                lim_row['SNID'] = snid
                lim_row['GALID_true'] = galid
                lim_row['X_WORLD'] = ra
                lim_row['Y_WORLD'] = dec
                lim_row['DLR'] = 0
                lim_row['ANGSEP'] = 0
                lim_row['DLR_RANK'] =0

                if lims:

                    res_df = res_df.append(lim_row)
                    res_df = res_df.append(lim_row)

                    res_df = res_df.drop(0)
                else:
                    res_df = res_df.append(lim_row)
            except:
                pass




        if type(res_df)==pd.DataFrame:
            res_df['EDGE_FLAG'] = get_edge_flags(res_df.X_IMAGE.values,res_df.Y_IMAGE.values)
        else:
            res_df['EDGE_FLAG'] = get_edge_flags(np.array([res_df.X_IMAGE]),np.array([res_df.Y_IMAGE]))[0]

        main_res_df = main_res_df.append(res_df)
    if add_lim:
        logger.debug('Is limit, setting dlr of host to 0')
        main_res_df.loc[ind,['DLR']] = 0
        rank = main_res_df['DLR'].rank(method='dense').astype(int)
    else:
        rank = main_res_df['DLR'].rank(method='dense').astype(int)

    for counter, r in enumerate(main_res_df['DLR'].values):
        if r >4:
            rank.iloc[counter]*=-1
    main_res_df['DLR_RANK']=rank
    if len(main_res_df[main_res_df['DLR']==0])>0:
        main_res_df.sort_values('DLR',inplace=True)
        main_res_df['DLR_RANK'] = main_res_df['DLR_RANK'] - (main_res_df['DLR_RANK']/np.abs(main_res_df['DLR_RANK']))
        main_res_df['DLR_RANK'].iloc[0] = 0

    main_res_df['SN_RA'] = ra
    main_res_df['SN_DEC'] = dec
    main_res_df.to_csv(resdir+'%s.result'%int(snid),index=True)

def get_DLR_ABT(RA_SN, DEC_SN, RA, DEC, A_IMAGE, B_IMAGE, THETA_IMAGE, angsep):
    '''Function for calculating the DLR of a galaxy - SN pair (taken from dessne)'''

    # inputs are arrays
    rad  = np.pi/180                   # convert deg to rad
    pix_arcsec = 0.264                 # pixel scale (arcsec per pixel)
    pix2_arcsec2 = 0.264**2            # pix^2 to arcsec^2 conversion factor
    pix2_deg2 = pix2_arcsec2/(3600**2) # pix^2 to deg^2 conversion factor
    global numFailed
    rPHI = np.empty_like(angsep)
    d_DLR = np.empty_like(angsep)

    # convert from IMAGE units (pixels) to WORLD (arcsec^2)
    A_ARCSEC = A_IMAGE*pix_arcsec
    B_ARCSEC = B_IMAGE*pix_arcsec

    # angle between RA-axis and SN-host vector
    GAMMA = np.arctan((DEC_SN - DEC)/(np.cos(DEC_SN*rad)*(RA_SN - RA)))

    # angle between semi-major axis of host and SN-host vector
    PHI = np.radians(THETA_IMAGE) + GAMMA # angle between semi-major axis of host and SN-host vector

    rPHI = A_ARCSEC*B_ARCSEC/np.sqrt((A_ARCSEC*np.sin(PHI))**2 +
                                     (B_ARCSEC*np.cos(PHI))**2)

    # directional light radius
    #  where 2nd moments are bad, set d_DLR = 99.99
    d_DLR = angsep/rPHI

    return [d_DLR, A_ARCSEC, B_ARCSEC, rPHI]

def get_edge_flags(xs,ys,dist=20):
    '''Flags objects that are near the edge of a chip'''

    flags = np.zeros_like(xs)
    for counter,x in enumerate(xs):
        if x<20 or x>4080:
            flags[counter]=1
    for counter,y in enumerate(ys):
        if y<20 or y>2080:
            flags[counter]=1
    return flags

def worker(args):

    galid,snid,ra,dec,resdir = [args[i]for i in range(len(args))]
    match_fakes(galid,snid,ra,dec,resdir,dist_thresh = 60,y=2,chip=21,f='SN-X3')
    return

def multi(fakes,resdir):

    pool_size = multiprocessing.cpu_count()*2
    act = multiprocessing.active_children()
    pool = multiprocessing.Pool(processes=pool_size,
                                maxtasksperchild=2,
                                )

    all_args = []
    for snrow in fakes.iterrows():

        sn = snrow[1]
        all_args.append([sn['GALID'],sn['ID'],sn['SN_RA'],sn['SN_DEC'],resdir])
    results=[]
    for _ in tqdm.tqdm(pool.imap_unordered(worker,all_args),total=len(all_args)):
        results.append(_)

    pool.close()
    pool.join()
    pool.close()
    return results

def main(fn,resdir = '/media/data3/wiseman/des/mismatch/fakes/',key=fakes):
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    fn_suffix = fn.split('.')[-1]
    if fn_suffix=='csv':
        fakes = pd.read_csv(fn,index_col=0)
    elif fn_suffix=='h5':
        fakes = pd.read_hdf(fn,key=key)
    multi(fakes,resdir)

    snlist = fakes['ID'].values
    with progressbar.ProgressBar(max_value=len(snlist)) as bar:
        for counter,sn in enumerate(snlist):
            res_fn = fn.replace('fakes','matched_fakes')
            res_fn = res_fn.replace(fn_suffix,'result')
            main_f = open(res_fn,'a')
            cat = os.path.join(resdir,'%s.result'%sn)
            c = open(cat,'r')
                #print ('Adding cat: %s'%cat, ' of length ',len(c.readlines()))
            for counter,l in enumerate(c.readlines()):
                if counter!=0:
                    main_f.write(l)
            main_f.close()
            bar.update(counter)
    return res_fn
if __name__=="__main__":
    main()
