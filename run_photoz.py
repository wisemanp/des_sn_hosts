import numpy as np
import pandas as pd
import argparse
import os
import yaml
import multiprocessing
import multiprocessing.pool
from multiprocessing.pool import ThreadPool
import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor

from des_sn_hosts.functions import photoz
from des_sn_hosts.utils.utils import get_good_des_chips, MyPool
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fields',help='Which DES fields to photo-z (comma-separated list): [X3]',default='X3',required=False)
    parser.add_argument('-ch','--chips', help='Which chips to use (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-my','--my',help='Which minusyear to use [2]',default='2',required=False)
    parser.add_argument('-c','--config',help='Path to config file [des_sn_hosts/config/config_photoz.yaml]',
                                                                    default='config/config_photoz.yaml')
    parser.add_argument('-fz','--fixz',help='Fix at already calculated redshifts',action='store_true')

    args = parser.parse_args()
    args.fields = args.fields.split(',')
    if args.chips == 'all':
        args.chips = get_good_des_chips()
    else:
        args.chips =args.chips.split(',')
    return args


def pz_worker(worker_args):
    #print('Made it to the pz worker ')
    args = worker_args[0]
    f = worker_args[1]
    ch = worker_args[2]

    args.output = os.path.join(args.config['des_root'],
        '5yr_stacks/MY%s/SN-%s/CAP/%s/'%(args.my,f,ch),
        '%s_SN-%s_%s_%s_%s_%s_%s'%(args.my,f,ch,args.config['cat_version'],args.config['params']['Z_MAX'],
        args.config['params']['Z_STEP'],args.config['params']['TEMPLATES_FILE'].split('/')[-2]))

    if not args.fixz:
        cat_fn = os.path.join(args.config['des_root'],
            '5yr_stacks/MY%s/SN-%s/CAP/%s/%s_SN-%s_%s_obj_deep_v%s.cat'%(args.my,f,ch,args.my,f,ch,args.config['cat_version']))
    else:
        cat_fn = args.output+'.eazypy.zout.h5'
        if not os.path.isfile(cat_fn):
            print('Cant find photoz file for CCD%s'%ch)
            return
        args.config['params']['TEMPLATES_FILE']='templates/fsps_full/tweak_fsps_QSF_12_v3.param'
        args.config['params']['FIX_ZSPEC'] = 'y'
        args.output = os.path.join(args.config['des_root'],
            '5yr_stacks/MY%s/SN-%s/CAP/%s/'%(args.my,f,ch),
            '%s_SN-%s_%s_%s_%s_%s_%s_fixedz'%(args.my,f,ch,args.config['cat_version'],args.config['params']['Z_MAX'],
            args.config['params']['Z_STEP'],args.config['params']['TEMPLATES_FILE'].split('/')[-2]))
    args.input = cat_fn
    print('Initializing EAZY on CCD %s'%ch)
    #print('Output name root: %s'%args.output)
    if not os.path.isfile(args.output +'.eazypy.zout.fits'):
        #print('Sending to photoz module!')
        photoz.main(args)
    else:
        print('Skipping CCD%s, already done!'%ch)
    return

def multi_pz(args,f):
    pool_size = 32
    #pool = ThreadPool(processes=pool_size)
    #pool = MyPool(processes=pool_size)
    pool = MyPool(processes=pool_size)
    #executor = ThreadPoolExecutor()
    worker_args = []
    for ch in args.chips:
        worker_args.append([args,f,ch])
    '''jobs = [executor.submit(pz_worker,[args,f,ch]) for ch in args.chips]
    for job in tqdm.tqdm(as_completed(jobs),total=len(args.chips)):
        pass'''
    print('Sending jobs to worker')

    results = pool.imap_unordered(pz_worker,worker_args)
    pool.close()
    pool.join()
def main():
    args = parser()
    config = yaml.load(open(args.config))
    args.config = config
    for f in args.fields:
        print('Initializing EAZY in the SN-%s field'%f)
        print('With this config:')
        print(args.config)
        multi_pz(args,f)

if __name__=="__main__":
    main()
