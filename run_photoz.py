import numpy as np
import pandas as pd
import argparse
import os
import yaml
from des_sn_hosts.functions import photoz
from des_sn_hosts.utils.utils import get_good_des_chips
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fields',help='Which DES fields to photo-z (comma-separated list): [X3]',default='X3',required=False)
    parser.add_argument('-ch','--chips', help='Which chips to use (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-my','--my',help='Which minusyear to use [2]',default='2',required=False)
    parser.add_argument('-c','--config',help='Path to config file [des_sn_hosts/config/config_photoz.yaml]',
                                                                    default='config/config_photoz.yaml')

    args = parser.parse_args()
    args.fields = args.fields.split(',')
    if args.chips == 'all':
        args.chips = get_good_des_chips()
    else:
        args.chips =args.chips.split(',')
    return args
def main():
    args = parser()
    config = yaml.load(open(args.config))
    args.config = config
    for f in args.fields:
        for ch in args.chips:
            cat_fn = os.path.join(config['des_root'],
                '5yr_stacks/MY%s/SN-%s/CAP/%s/%s_SN-%s_%s_obj_deep_v%s.cat'%(args.my,f,ch,args.my,f,ch,config['cat_version']))
            args.input = cat_fn
            args.output = '%s_%s_%s_%s_%s_%s'%(args.my,f,ch,config['cat_version'],config['params']['Z_MAX'],config['params']['Z_STEP'])
            photoz.main(args)
if __name__=="__main__":
    main()
