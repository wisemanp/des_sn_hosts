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
    parser.add_argument('-sn','--sn_fn',help='SN Hosts filename',default='/media/data3/wiseman/des/desdtd/data/5yr_MV_20200701_FITRES.csv')
    parser.add_argument('-fi','--field_fn',help='Field filename',default='/media/data3/wiseman/des/coadding/results/deep/all_fields_MY2_photoz.csv')
    parser.add_argument('-c','--config',help='Config filename',default='/home/wiseman/code/des_sn_hosts/config/config_rates.yaml')
    parser.add_argument('-f','--fields',help='DES fields to use',default=None)
    parser.add_argument('-zc','--zmax',help='Maximum redshift',default=None)
    args = parser.parse_args()
    return args

def main():
    args=parser()
    print('Parsed args, going to set up Rates instance')
    r = Rates(config=yaml.load(open(args.config)),
                SN_hosts_fn = args.sn_fn,
                field_fn = args.field_fn,
                fields=args.fields)
    r.field['mass'] = np.log10(r.field['mass'])
    r.field['mass_err']=0.3
    r.field['SFR'] = np.log10(r.field['SFR'])
    r.field['SFR_err']=0.3
    r.field['ssfr'] = r.field['SFR'] - r.field['mass']
    r.field['ssfr_err'] = 0.4
    r.SN_Hosts['logssfr'] = r.SN_Hosts['logsfr']-r.SN_Hosts['HOST_LOGMASS']
    r.SN_Hosts['logssfr_err'] =(r.SN_Hosts['logsfr_err']**2 + r.SN_Hosts['HOST_LOGMASS_ERR']**2)**0.5
    r.get_SN_bins()

    r.get_field_bins()
    print('Going to generate SN resamples')
    r.generate_sn_samples(n_iter=int(1E+3))
    print('Going to generate field resamples')
    r.generate_field_samples(n_iter=int(1E+2))
    print('Going to sample the rate from the resampled data!')
    r.SN_G_MC(n_samples=100)
    print('Going to fit the rates')
    fit = r.fit_SN_G(n_iter=4000)
if __name__=="__main__":
    main()
