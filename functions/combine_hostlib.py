import pandas as pd
import numpy as np
import os
import argparse
import yaml
from astropy.table import Table
from des_stacks.utils.gen_tools import get_good_des_chips
good_des_chips = get_good_des_chips()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fields',help='Which DES fields to photo-z (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-ch','--chips', help='Which chips to use (comma-separated list): [all]',default='all',required=False)
    parser.add_argument('-my','--my',help='Which minusyear to use [2]',default='2',required=False)
    parser.add_argument('-c','--config',help='Path to config file [des_sn_hosts/config/config_photoz.yaml]',
                                                                    default='config/config_photoz.yaml')
    parser.add_argument('-sf','--savename',help='Name of saved file',default=None)
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

    if args.chips !='all':
        chips = args.chips.split(',')
    else:
        args.chips = good_des_chips
    main_df = pd.DataFrame()
    if args.df !='none':
        main_df = pd.read_csv(args.df,index_col=0)
    config = yaml.load(open(args.config,'r'))
    for my in mys:

        for f in fields:

            for ch in good_des_chips:
                ch = int(ch)
                fn = os.path.join(config['des_root'],
                    '5yr_stacks/MY%s/SN-%s/CAP/%s/'%(args.my,f,ch),
                    '%s_SN-%s_%s_%s_%s_%s_%s'%(args.my,f,ch,config['cat_version'],config['params']['Z_MAX'],config['params']['Z_STEP'],config['params']['TEMPLATES_FILE'].split('/')[-2]))
                if 1==1:
                    if os.path.isfile(fn+'.eazypy.zout.fits'):
                        zphot_res = Table.read(fn+'.eazypy.zout.fits')
                        zphot_res.remove_columns(['Avp','massp','SFRp','sSFRp','LIRp'])
                        zphot_res = zphot_res.to_pandas()
                        print ('Adding cat: %s.eazypy.zout.fits'%fn, ' of length ',len(zphot_res))
                        in_fn = fn+'.cat'
                        zphot_in = Table.read(in_fn,format='ascii').to_pandas()
                        zphot_res = zphot_res.merge(zphot_in,on='id',how='outer')
                        snf = 'SN-'+f

                        cat_deep = pd.read_csv(os.path.join('/media/data3/wiseman/des/coadding/5yr_stacks/MY%s'%args.my,snf,'CAP',str(ch),
                            '%s_%s_%s_obj_deep_v7.cat'%(args.my,snf,ch)))
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
                        cat_deep.replace(-9999.000000,np.NaN,inplace=True)
                        cat_deep.dropna(subset=['MAGERR_AUTO_R'],inplace=True)
                        cat_deep.index = cat_deep.index.astype(int)
                        cat_deep['id'] = cat_deep.index.values
                        cat_deep = cat_deep[(cat_deep['X_IMAGE']>200)&(cat_deep['X_IMAGE']<4200)&(cat_deep['Y_IMAGE']>80)&(cat_deep['Y_IMAGE']<2080)]

                        cat_deep = cat_deep.merge(zphot_res,on='id',how='outer')


                        cat_deep.drop([
                                'FLUX_RADIUS_g','FLUX_RADIUS_r','FLUX_RADIUS_i','FLUX_RADIUS_z',
                               'FWHM_WORLD_g','FWHM_WORLD_r','FWHM_WORLD_i','FWHM_WORLD_z'],axis=1,inplace=True)



                        cat_deep.to_hdf(fn+'.eazypy.zout.h5')     
                        main_df = main_df.append(cat_deep)


                    else:
                        print('Missing %s'%fn)
    if not args.savename:
        args.savename = '%s_%s_photoz.h5'%(fields,args.my)
    main_df.to_hdf(os.path.join(config['des_root'],'results','deep',args.savename),key='photoz')
    print ('Saved new file to ',os.path.join(config['des_root'],'results','deep',args.savename))
if __name__=="__main__":
    args=parser()
    main(args)
