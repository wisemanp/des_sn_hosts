import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse

from des_sn_hosts.rates.rates import Rates
import yaml


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn','--sn_fn',help='SN Hosts filename',default='/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701.FITRES')
    parser.add_argument('-fi','--field_fn',help='Field filename',default='/media/data3/wiseman/des/coadding/results/deep/all_2_photoz.csv')
    parser.add_argument('-c','--config',help='Config filename',default='/home/wiseman/code/des_sn_hosts/config/config_rates.yaml')
    parser.add_argument('-f','--fields',help='DES fields to use',default=None)

def main():
    args=parser()
    r = Rates(args.sn_fn,args.field_fn,
              config=yaml.load(open(args.config)),fields=['C1','C2','C3'])
    r.field['mass'] = np.log10(r.field['mass'])
    r.field['mass_err']=0.1
    r.get_SN_bins()

    r.get_field_bins()
    r.generate_sn_samples(n_iter=int(1E+3))

    r.generate_field_samples(n_iter=int(1E+2))


if __name__=="__main__":
    main()
