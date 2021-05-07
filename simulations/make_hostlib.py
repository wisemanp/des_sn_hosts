import numpy as np
import pandas as pd
from tqdm import tqdm 
from dust_extinction.parameter_averages import F19
from spectral_utils import load_spectrum, convert_escma_fluxes_to_griz_mags
from synspec import SynSpec
import argparse
bc03_flux_conv_factor = 3.12e7
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z','--z',help='Redshift',default=0.5,type=str)
    parser.add_argument('-al','--av_lo',help='Lowest Av',default=0,type=float)
    parser.add_argument('-ah','--av_hi',help='Highest Av',default=1,type=float)
    parser.add_argument('-na','--n_av',help='Av step',default=20,type=float)
    parser.add_argument('-at','--av_step_type',help='Av step type (lin or log)',default='lin')
    parser.add_argument('-u','--logU',help='Ionisation parameter',default=-2,type=float)
    parser.add_argument('-tr','--time_res',help='SFH time resolution',default=5,type=int)

    args = parser.parse_args()
    return args

def run(args):
    # DES filter objects

    filt_dir = '/media/data3/wiseman/des/AURA/filters/'
    filt_obj_list = [
        load_spectrum(filt_dir+'decam_g.dat'),
        load_spectrum(filt_dir+'decam_r.dat'),
        load_spectrum(filt_dir+'decam_i.dat'),
        load_spectrum(filt_dir+'decam_z.dat'),
    ]

    nfilt = len(filt_obj_list)

    aura_dir = '/media/data3/wiseman/des/AURA/'
    #------------------------------------------------------------------------
    # BC03 SSPs as mc_spec Spectrum objects
    f1 = open(aura_dir+'/bc03_logt_list.dat')
    bc03_logt_list = [x.strip() for x in f1.readlines()]
    f1.close()
    bc03_logt_array = np.array(bc03_logt_list)
    ntemp = len(bc03_logt_array)

    bc03_dir = '/media/data1/childress/des/galaxy_sfh_fitting/bc03_ssp_templates/'
    template_obj_list = []
    nLy_list = []
    for i in range(ntemp):
        bc03_fn = '%sbc03_chabrier_z02_%s.spec' % (bc03_dir, bc03_logt_list[i])
        new_template_spec =  load_spectrum(bc03_fn)
        template_obj_list.append(new_template_spec)
    s = SynSpec(template_obj_list = template_obj_list,neb=True)
    store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_alt_0.5_Qerf_1.1.h5','r')
    ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])
    results = []
    z_array = [float(z) for z in args.z.split(',')]
    if args.av_step_type == 'lin':
        av_arr = np.linspace(args.av_lo,args.av_hi,args.n_av)
    elif args.av_step_type == 'log':
        av_arr = np.logspace(args.av_lo,args.av_hi,args.n_av)
    for z in z_array:
        distance_factor = 10.0**(0.4*cosmo.distmod(z).value)
        for tf in tqdm(ordered_keys[::-1][np.arange(0,len(ordered_keys),args.time_res)]):   # Iterate through the SFHs for galaxies of different final masses
            sfh_df = store['/'+str(tf)]
            sfh_df = sfh_df[sfh_df['z']>z]
            if len(sfh_df)>0:
                mtot=sfh_df['m_tot'].iloc[-1]
                ssfr = np.sum(sfh_df['m_formed'].iloc[-500:])/((250*1E+6)*mtot)
                sfr = ssfr*mtot
                sfh_df['stellar_age'] = sfh_df.age.values[::-1]
                dtd_x1hi = phi_t_pl(sfh_df['stellar_age']/1000,0.04,beta_x1hi,norm_x1hi)
                pred_rate_x1hi =np.sum(sfh_df['m_formed']*dtd_x1hi)
                dtd_x1lo = phi_t_pl(sfh_df['stellar_age']/1000,0.04,beta_x1lo,norm_x1lo)
                pred_rate_x1lo =np.sum(sfh_df['m_formed']*dtd_x1lo)
                dtd_total =phi_t_pl(sfh_df['stellar_age']/1000,0.04,beta,dtd_norm)
                SN_age_dist = sfh_df['m_formed']*dtd_total
                pred_rate_total = np.sum(SN_age_dist)
                ages = sfh_df['stellar_age']/1000
                mwsa = np.average(sfh_df['stellar_age'],weights=sfh_df['m_formed']/mtot)
                sfh_coeffs_PW21 = interpolate_SFH(sfh_df,mtot)
                if mtot>1E+10:
                    mu_Rv = 2.6
                    avs_SBL =np.clip(np.random.normal(av_means_mhi(np.log10(mtot)),av_sigma(np.log10(mtot)),size=20),a_min=0,a_max=None)
                else:
                    mu_Rv = 3.1
                    avs_SBL = np.clip(np.random.normal(av_means_mlo,av_sigma(np.log10(mtot)),size=20),a_min=0,a_max=None)
                for Av in av_arr:
                    Rv = np.min([np.max([2.0,np.random.normal(mu_Rv,0.5)]),6.0])
                    delta='None'
                    U_R,fluxes= s.calculate_model_fluxes_pw(sfh_coeffs_PW21,z=z,dust={'Av':Av,'Rv':Rv,'delta':'none','law':'CCM89'},neb=True,logU=args.logU)
                    obs_flux = mtot*fluxes/(distance_factor*bc03_flux_conv_factor)
                    results.append(np.concatenate([[z,mtot,ssfr,mwsa,Av,Rv,delta,U_R[0],pred_rate_x1hi,pred_rate_x1lo,ages,SN_age_dist,pred_rate_total],obs_flux[0]]))

    flux_df = pd.DataFrame(results,columns=['z','mass','ssfr','mean_age','Av','Rv','delta','U_R','pred_rate_x1_hi','pred_rate_x1_lo','SN_ages','SN_age_dist','pred_rate_total','f_g','f_r','f_i','f_z'])
    zp_fluxes = np.array([2.207601113629584299e-06,1.880824499994395390e-06,1.475307638780991749e-06,1.014740352137762549e-06])
    default_des_syserrs = np.array([0.03, 0.02, 0.02, 0.03])
    mags,fuJys=convert_escma_fluxes_to_griz_mags(flux_df[['f_g','f_r','f_i','f_z']])
    flux_df[['f_g','f_r','f_i','f_z',]] =fuJys
    flux_df[['mag_g','mag_r','mag_i','mag_z']]=mags
    flux_df['g_r'] = flux_df['mag_g'] - flux_df['mag_r']
    flux_df.to_hdf('/media/data3/wiseman/des/AURA/all_model_params_z%.2f_%.2f_av%.2f_%.2f_rv_rand_full_age_dists_neb_U%.2f_res_%i.h5'%(z_array[0],z_array[-1],av_arr[0],av_arr[-1],args.logU,args.time_res),key='main')
    print("Done!")
if __name__=="__main__":
    args = parser()
    run(args)