import numpy as np
import pandas as pd
import subprocess
import glob
import matplotlib.pyplot as plt
import os
import astropy
import sys
import eazy
import time
eazy.symlink_eazy_inputs()
eazy_dir = os.getenv('EAZYCODE')
import warnings
from astropy.utils.exceptions import AstropyWarning

from des_sn_hosts.utils.utils import get_edge_flags
np.seterr(all='ignore')
warnings.simplefilter('ignore')

bands=['g','r','i','z']

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',help='Catalog file to run photo-zs on',required=True)
    parser.add_argument('-o','--output',help='Name to give to the eazy output files',required=True)
    parser.add_argument('-c','--config',help='Config file path',required=False,default='config/config_photoz.yaml')
    return parser.parse_args()

def prep_eazy_data(allgals,args):

    allgals = allgals[['ID','SPECZ','Y_IMAGE',
                       'RA','DEC',
                       'MAG_AUTO_G','MAGERR_STATSYST_AUTO_G',
                       'MAG_AUTO_R','MAGERR_STATSYST_AUTO_R',
                       'MAG_AUTO_I','MAGERR_STATSYST_AUTO_I',
                       'MAG_AUTO_Z','MAGERR_STATSYST_AUTO_Z',
                       'FLUX_AUTO_G','FLUXERR_AUTO_G',
                       'FLUX_AUTO_R','FLUXERR_AUTO_R',
                       'FLUX_AUTO_I','FLUXERR_AUTO_I',
                       'FLUX_AUTO_Z','FLUXERR_AUTO_Z',
                       'MAG_ZEROPOINT_G','MAG_ZEROPOINT_ERR_G',
                       'MAG_ZEROPOINT_R','MAG_ZEROPOINT_ERR_R',
                       'MAG_ZEROPOINT_I','MAG_ZEROPOINT_ERR_I',
                       'MAG_ZEROPOINT_Z','MAG_ZEROPOINT_ERR_Z']]
    print ('Going to work on %s galaxies!'%len(allgals))

    for b in bands:
        allgals['FLUX_AUTO_uJy_%s'%b] = ''
        allgals['FLUX_AUTO_uJy_%s'%b] = ''
        allgals['FLUX_AUTO_uJy_%s'%b] =(10**6)*10**(3.56-(allgals['MAG_ZEROPOINT_%s'%b.capitalize()]/2.5))*allgals['FLUX_AUTO_%s'%b.capitalize()]
        allgals['FLUXERR_AUTO_uJy_%s'%b] = allgals['FLUX_AUTO_uJy_%s'%b].values*((2.303*allgals['MAG_ZEROPOINT_ERR_%s'%b.capitalize()]/2.5)**2 +\
    (allgals['FLUXERR_AUTO_%s'%b.capitalize()]/allgals['FLUX_AUTO_%s'%b.capitalize()])**2)**0.5
    for b in ['g','r','i','z']:
        allgals.loc[allgals[allgals['MAGERR_STATSYST_AUTO_%s'%b.capitalize()]<0].index,'FLUX_AUTO_uJy_%s'%b] =-1
    for_eazy = allgals.rename(columns={'ID':'id',
                                         'SPECZ':'redshift',
                                         'FLUX_AUTO_uJy_g':'decam_g',
                                         'FLUX_AUTO_uJy_r':'decam_r',
                                         'FLUX_AUTO_uJy_i':'decam_i',
                                         'FLUX_AUTO_uJy_z':'decam_z',
                                         'FLUXERR_AUTO_uJy_g':'decam_g_err',
                                         'FLUXERR_AUTO_uJy_r':'decam_r_err',
                                         'FLUXERR_AUTO_uJy_i':'decam_i_err',
                                         'FLUXERR_AUTO_uJy_z':'decam_z_err'
                                       })

    for_eazy = for_eazy[['id','redshift','decam_g','decam_r','decam_i','decam_z',
                             'decam_g_err','decam_r_err','decam_i_err','decam_z_err']]
    for_eazy.drop_duplicates('id',inplace=True)
    #for_eazyale.to_csv('/media/data3/wiseman/des/cigale/cigale-v2018.0/pcigale/confused_hosts.dat',sep=' ',index=False)
    for_eazy.replace(-9.998,-1,inplace=True)
    for_eazy.replace(np.NaN,-1,inplace=True)
    for_eazy.rename(columns = {'id':'# id',
                              },inplace=True)

    for_eazy.loc[0] =['# id', 'z_spec', 'F294', 'F295', 'F296', 'F297', 'E294', 'E295',
           'E296', 'E297']
    for_eazy.sort_index(inplace=True)
    print('Working path: %s'%args.output)
    if os.path.isdir(os.path.split(args.output)[0]):
        input_fn = '%s.cat'%args.output
    else:
        input_fn = os.path.join(os.getenv('EAZYCODE'),'outputs/%s.cat'%args.output)
    catfile = open(input_fn,'w')

    reshead_line0 = ''
    for i in for_eazy.columns:
        reshead_line0 +='\t'+ i
    reshead_line0 = reshead_line0[1:]

    reshead_line1 = ''
    for i in for_eazy.loc[0].values:
        reshead_line1 +='\t'+ i
    reshead_line1 = reshead_line1[1:]


    for_eazy.drop(0,inplace=True)

    for_eazy.to_csv(input_fn.replace('cat','tempcat'),sep='\t',index=False,header=False)
    stringthing = open(input_fn.replace('cat','tempcat'),'r')
    psfstring = stringthing.read()
    stringthing.close()
    #reshead = reshead_line0+'\n'
    reshead = reshead_line1+'\n'

    catfile.write(reshead)
    catfile.write(psfstring)
    catfile.close()
    return for_eazy

def run_eazy(args):
    params = args.config['params']

    if os.path.isdir(os.path.split(args.output)[0]):
        params['MAIN_OUTPUT_FILE'] = '%s.eazypy'%args.output
        params['CATALOG_FILE'] = '%s.cat'%args.output
    else:
        params['MAIN_OUTPUT_FILE'] = os.path.join(os.getenv('EAZYCODE'),'outputs/%s.eazypy'%args.output)
        params['CATALOG_FILE'] = os.path.join(os.getenv('EAZYCODE'), 'inputs/%s.cat'%args.output)
    translate_file = os.path.join(os.getenv('EAZYCODE'), 'inputs/zphot.translate')
    ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None,
                          params=params, load_prior=True, load_products=False)
    NITER = 3
    NBIN = np.minimum(ez.NOBJ//100, 180)

    ez.param.params['VERBOSITY'] = 1.
    #for iter in range(NITER):
    #    print('Iteration: ', iter)
    #    sn = ez.fnu/ez.efnu
    #    clip = (sn > 10).sum(axis=1) > 3 # Generally make this higher to ensure reasonable fits
    #    ez.iterate_zp_templates(idx=ez.idx[clip], update_templates=False,
    #                              update_zeropoints=True, iter=iter, n_proc=8,
    #                              save_templates=False, error_residuals=(iter > 0),
    #                              NBIN=NBIN, get_spatial_offset=False)
    # Turn off error corrections derived above
    ez.efnu = ez.efnu_orig*1

    # Full catalog
    sample = np.isfinite(ez.cat['z_spec'])

    ez.fit_parallel(ez.idx[sample], n_proc=16)
    zout, hdu = ez.standard_output(rf_pad_width=0.5, rf_max_err=2,
                                     prior=True, beta_prior=True)
    print('Saved results to %s'%(params['MAIN_OUTPUT_FILE']+'.zout'))
def main(args):
    start_time = float(time.time())
    df = pd.read_csv(args.input)

    df = df.rename(index=str,columns={'Unnamed: 0':'ID',
                                      'X_WORLD':'RA','Y_WORLD':'DEC',
                                        'MY':'SEASON',
                                        'ANGSEP':'SEPARATION',
                                       'z':'SPECZ',
                                       'ez':'SPECZ_ERR',
                                       'source':'SPECZ_CATALOG',
                                       'flag':'SPECZ_FLAG',
                                       #'objtype_ozdes':'OBJTYPE_OZDES',
                                       #'transtype_ozdes':'TRANSTYPE_OZDES',
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
                                       'MAGERR_STATSYST_AUTO_z':'MAGERR_STATSYST_AUTO_Z',
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
                                       'MAG_ZEROPOINT_ERR_z':'MAG_ZEROPOINT_ERR_Z'})

    df[pd.isna(df['MAGERR_STATSYST_AUTO_G'])]['MAGERR_AUTO_G'].unique()

    df['Z_RANK'] = df['Z_RANK'].fillna(-9.999)

    df['EDGE_FLAG'] = get_edge_flags(df.X_IMAGE.values,df.Y_IMAGE.values)

    df = df[(df['EDGE_FLAG']==0)& (df['Z_RANK']<2)]

    for b in ['G','R','I','Z']:
        df.loc[df[df['MAGERR_AUTO_%s'%b]<0].index,'MAGERR_STATSYST_AUTO_%s'%b]=-1
        df['MAGERR_STATSYST_AUTO_%s'%b].replace(np.NaN,-1)
        df['MAGERR_STATSYST_AUTO_%s'%b] = df['MAGERR_STATSYST_AUTO_%s'%b].astype(float)

    df.index = df.index.astype(int)
    df['ID'] = df.index.values
    #df =df[df['SPECZ']>0]
    prep_eazy_data(df,args)
    run_eazy(args)
    t = float(time.time()) - start_time
    print('Took %.1f seconds'%t)
if __name__ == "__main__":
    args = parser()
    main(args)
