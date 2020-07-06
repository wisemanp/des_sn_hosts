import pandas as pd
import numpy as np
import os
import argparse
from des_stacks.utils.gen_tools import get_good_des_chips
good_des_chips = get_good_des_chips()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fields',help='Which DES fields to photo-z (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-ch','--chips', help='Which chips to use (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-my','--my',help='Which minusyear to use [2]',default='2',required=False)
    parser.add_argument('-c','--config',help='Path to config file [des_sn_hosts/config/config_photoz.yaml]',
                                                                    default='config/config_photoz.yaml')
    parser.add_argument('-df','--df',default='none')

    return parser.parse_args()

def main(args):

    if args.fields =='all':
        fields =['C1','C2','C3','X1','X2','X3','S1','S2','E1','E2']
    else:
        fields = args.fields.split(',')
    mys = ['none']

    if args.my !='none':
        mys = args.my

    if args.chip !='all':
        chips = args.chip.split(',')
    else:
        args.chip good_des_chips
    main_df = pd.DataFrame()
    if args.df !='none':
        main_df = pd.read_csv(args.df,index_col=0)
    for my in mys:

        for f in fields:

            for ch in good_des_chips:
                ch = int(ch)
                fn = os.path.join(args.config['des_root'],
                    '5yr_stacks/MY%s/SN-%s/CAP/%s/'%(args.my,f,ch),
                    '%s_SN-%s_%s_%s_%s_%s'%(args.my,f,ch,args.config['cat_version'],args.config['params']['Z_MAX'],args.config['params']['Z_STEP']))
                cat_df = pd.read_csv(fn,index_col=0)
                print ('Adding cat: %s'%cat, ' of length ',len(cat_df))
                main_df = main_df.append(cat_df)
    main_df.to_csv(os.path.join(args.config['des_root'],'results','deep','%s_%s_photoz.csv'%(args.fields,args.my),index=True)
    print ('Saved new file to %s'%os.path.join(args.config['des_root'],'results','deep','%s_%s_photoz.csv'%(args.fields,args.my))
if __name__=="__main__":
    args=parser()
    main(args)
