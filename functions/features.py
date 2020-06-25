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
from tqdm.contrib.concurrent import process_map
import pandas as pd
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import itertools
import warnings
warnings.simplefilter('ignore')

def compute_features(DLR,SEP):
    dlr = np.sort(DLR) # first sort DLR array
    sep = np.sort(SEP)
    e = 1e-5 # define epsilon, some small number
    if len(dlr)==1: # only 1 object in radius
        HC = -99.0
        D12 = D1_D2 = D13 = D1_D3 = S12 = S12_DLR = S1_S2 = S13 = S1_S3 = 0
    else:
        # closest match
        delta12 = dlr[1] - dlr[0] + e
        D12 =dlr[1] - dlr[0]
        D1D2 = dlr[0]**2/dlr[1] + e
        D1_D2 = dlr[0]/dlr[1]
         # sep
        DLR_sorted_SEP = sep[np.argsort(DLR)]
        S12_DLR = DLR_sorted_SEP[1] - DLR_sorted_SEP[0]
        S12 = sep[1] - sep[0]
        S1_S2 = sep[0]/sep[1]

        D13 = dlr[2] - dlr[0]
        D1_D3 = dlr[0]/dlr[2]
        S13 = sep[2] - sep[0]
        S1_S3 = sep[0]/sep[2]
        Dsum = 0
        for i in range(0, len(dlr)):
            for j in range(i+1, len(dlr)):
                didj = dlr[i]/dlr[j] + e
                delta_d = dlr[j] - dlr[i] + e
                Dsum += didj/((i+1)**2*delta_d)

        HC = np.log10(D1D2*Dsum/delta12)
    return HC, D12, D1_D2, D13, D1_D3, S12, S12_DLR, S1_S2, S13, S1_S3

def worker(g):
    feature_cols = ['HC','D12','D1D2','D13','D1D3','S12','S12_DLR','S1S2','S13','S1S3']
    if len(g[(g['DLR']>0)&(g['Z_RANK']<2)])>2:
        host_candidates = g[(g['DLR']>0)&(g['Z_RANK']<2)]

        g[feature_cols] = compute_features(host_candidates['DLR'].values,host_candidates['ANGSEP'].values)
    elif len(g[g['DLR']==0])==1:
        host_candidates = g[(g['DLR']>0)&(g['Z_RANK']<2)]
        if len(g[(g['DLR']>0)&(g['Z_RANK']<2)])>2:
            g[feature_cols] = compute_features(host_candidates['DLR'].values,host_candidates['ANGSEP'].values)
        else:
            g[feature_cols]=99

    else:
        g[feature_cols]=99
    return g

def multi(matched_fakes,key='fakes'):
    pool_size = multiprocessing.cpu_count()*2
    pool = multiprocessing.Pool(processes=pool_size,
                                maxtasksperchild=2,
                                )
    snidgroups = matched_fakes.groupby('SNID')
    feature_cols = ['HC','D12','D1D2','D13','D1D3','S12','S12_DLR','S1S2','S13','S1S3']
    matched_fakes[feature_cols] = pd.DataFrame(columns=feature_cols,index=matched_fakes.index)
    all_args = []
    results = tqdm.tqdm(pool.imap_unordered(worker,[g for n,g in snidgroups]),total=len(snidgroups))
    return pd.concat(results)

def main(fn='/media/data3/wiseman/des/mismatch/matched_fakes.csv',key='fakes'):
    fn_suffix = fn.split('.')[-1]
    if fn_suffix=='csv':
        matched_fakes = pd.read_csv(fn,index_col =0)
    elif fn_suffix=='h5':
        print('Attempting to read in %s'%fn)
        matched_fakes = pd.read_hdf(fn,key=key)
    print('Calculating features for the host matches...')
    matched_fakes = multi(matched_fakes)
    print('Done calculating features!')
    features_fn = fn.replace('_matched_fakes','_matched_fakes_features')
    print('Saving to %s'%features_fn)
    matched_fakes.to_hdf(features_fn,key=key)
    return features_fn,matched_fakes
if __name__=="__main__":
    main()
