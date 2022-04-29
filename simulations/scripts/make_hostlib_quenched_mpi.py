import numpy as np
import pandas as pd
from tqdm import tqdm
from dust_extinction.parameter_averages import F19
from des_sn_hosts.simulations.spectral_utils import load_spectrum, convert_escma_fluxes_to_griz_mags,interpolate_SFH,interpolate_SFH_pegase
from des_sn_hosts.simulations.synspec import SynSpec, phi_t_pl
from des_sn_hosts.utils.utils import MyPool
import argparse
from astropy.cosmology import FlatLambdaCDM
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
cosmo = FlatLambdaCDM(70,0.3)
bc03_flux_conv_factor = 3.12e7

# DTD parameters from W21
beta_x1hi = -1.68
norm_x1hi = 0.51E-13
beta_x1lo = -0.79
norm_x1lo = 1.19E-13
beta = -1.14
#beta=-1.5
dtd_norm = 2.08E-13

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z','--z',help='Redshift',default=0.5,type=str)
    parser.add_argument('-zl','--zlo',help='Redshift lower end',default=0.15,type=float)
    parser.add_argument('-zh','--zhi',help='Redshift upper end',default=1.25,type=float)
    parser.add_argument('-zs','--zstep',help='Redshift step',default=0.15,type=float)
    parser.add_argument('-al','--av_lo',help='Lowest Av',default=0,type=float)
    parser.add_argument('-ah','--av_hi',help='Highest Av',default=1,type=float)
    parser.add_argument('-na','--n_av',help='Av step',default=20,type=int)
    parser.add_argument('-at','--av_step_type',help='Av step type (lin or log)',default='lin')
    parser.add_argument('-u','--logU',help='Ionisation parameter',default=-2,type=float)
    parser.add_argument('-tr','--time_res',help='SFH time resolution',default=5,type=int)
    parser.add_argument('-t','--templates',help='Template library to use [BC03, PEGASE]',default='BC03',type=str)
    parser.add_argument('-tf','--templates_fn',help='Filename of templates',type=str,default='None')
    parser.add_argument('-ne','--neb',action='store_true')
    parser.add_argument('-b','--beta',help='Absolute value of the slope of the DTD',default=1.14,type=float)
    args = parser.parse_args()
    return args

def sed_worker(worker_args):
    sfh_df,args,av_arr,z,tf,s,bc03_logt_float_array = [worker_args[i] for i in range(7)]
    results = []
    for i in tqdm(sfh_df.index.unique()):
        sfh_iter_df = sfh_df.loc[i]
        mtot=sfh_iter_df['m_tot'].iloc[-1]
        age = sfh_iter_df['age'].iloc[-1]
        #print('Mass: ',np.log10(mtot),'age: ',age)
        ssfr = np.sum(sfh_iter_df['m_formed'].iloc[-500:])/((250*1E+6)*mtot)
        sfr = ssfr*mtot
        sfh_iter_df['stellar_age'] = sfh_iter_df.age.values[::-1]
        ages = sfh_iter_df['stellar_age']/1000
        dtd_x1hi = phi_t_pl(ages,0.04,beta_x1hi,norm_x1hi)
        pred_rate_x1hi =np.sum(sfh_iter_df['m_formed']*dtd_x1hi)
        dtd_x1lo = phi_t_pl(ages,0.04,beta_x1lo,norm_x1lo)
        pred_rate_x1lo =np.sum(sfh_iter_df['m_formed']*dtd_x1lo)
        dtd_total =phi_t_pl(ages,0.04,-1*args.beta,dtd_norm)
        SN_age_dist = sfh_iter_df['m_formed']*dtd_total
        pred_rate_total = np.sum(SN_age_dist)

        mwsa = np.average(sfh_iter_df['stellar_age'],weights=sfh_iter_df['m_formed']/mtot)
        if np.log10(mtot)<=9.5:
            mu_Rv = 2.61
        elif 9.5 <np.log10(mtot)<=10.5:
            mu_Rv = 2.99
            #avs_SBL =np.clip(np.random.normal(av_means_mhi(np.log10(mtot)),av_sigma(np.log10(mtot)),size=20),a_min=0,a_max=None)
        else:
            mu_Rv = 3.47
            #avs_SBL = np.clip(np.random.normal(av_means_mlo,av_sigma(np.log10(mtot)),size=20),a_min=0,a_max=None)
        if args.templates == 'BC03':
            sfh_coeffs_PW21 = interpolate_SFH(sfh_iter_df,mtot,bc03_logt_float_array)
            template=None
        elif args.templates =='PEGASE':
            if args.templates_fn =='None':
                templates = pd.read_hdf('/media/data3/wiseman/des/AURA/PEGASE/templates.h5',key='main')
            else:
                templates = pd.read_hdf(args.templates_fn,key='main')
            sfh_coeffs_PW21 = interpolate_SFH_pegase(sfh_iter_df,templates['time'],mtot,templates['m_star'])
        arr = np.zeros((len(ages),2))
        arr[:,0] = ages
        arr[:,1] = SN_age_dist
        np.savetxt('/media/data3/wiseman/des/AURA/sims/hostlibs/SN_ages/all_model_params_quench_%s_z_%.2f_rv_rand_full_age_dists_neb_U%.2f_res_%i_beta_%.2f_%.1f_%i.dat'%(args.templates,z,args.logU,args.time_res,args.beta,tf,i),arr)
        for Av in av_arr:
            Rv = np.min([np.max([2.0,np.random.normal(mu_Rv,0.5)]),6.0])
            delta='None'
            #if args.templates =='PEGASE':
            #    sfh_coeffs_PW21 = None
            #    template = pd.read_hdf('/media/data3/wiseman/des/AURA/PEGASE/templates_analytic_orig_%i.h5' % tf,
            #                           key='main')
            U_R,fluxes,colours= s.calculate_model_fluxes_pw(z,sfh_coeffs_PW21,dust={'Av':Av,'Rv':Rv,'delta':'none','law':'CCM89'},
                                                    neb=args.neb,logU=args.logU,mtot=mtot,age=age)
            obs_flux  = list(fluxes.values())#+cosmo.distmod(z).value
            U,B,V,R,I = (colours[i] for i in colours.keys())

            results.append(np.concatenate([[z,mtot,ssfr,mwsa,Av,Rv,delta,U_R[0],pred_rate_x1hi,pred_rate_x1lo,pred_rate_total],obs_flux[0],obs_flux[1],obs_flux[2],obs_flux[3],U,B,V,R,I,tf]))

    df = pd.DataFrame(results,columns=['z','mass','ssfr','mean_age','Av','Rv','delta','U_R','pred_rate_x1_hi',
                                            'pred_rate_x1_lo','pred_rate_total',
                                            'm_g','m_r','m_i','m_z','U','B','V','R','I','t_f'])
    #df['g_r'] = df['m_g'] - df['m_r']
    return df


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
    if args.templates =='BC03':
        bc03_logt_list = [x.strip() for x in f1.readlines()]
        f1.close()
        bc03_logt_array = np.array(bc03_logt_list)
        ntemp = len(bc03_logt_array)
        bc03_logt_float_array =np.array([float(x) for x in (bc03_logt_array)])
        bc03_dir = '/media/data1/childress/des/galaxy_sfh_fitting/bc03_ssp_templates/'
        template_obj_list = []
        nLy_list = []
        for i in range(ntemp):
            bc03_fn = '%sbc03_chabrier_z02_%s.spec' % (bc03_dir, bc03_logt_list[i])
            new_template_spec =  load_spectrum(bc03_fn)
            template_obj_list.append(new_template_spec)

        s = SynSpec(template_obj_list = template_obj_list,neb=args.neb)
        neb=args.neb
    elif args.templates=='PEGASE':
        s = SynSpec(library='PEGASE',template_dir = '/media/data3/wiseman/des/AURA/PEGASE/templates/',neb=args.neb)

        neb=args.neb
    store = pd.HDFStore('/media/data3/wiseman/des/desdtd/SFHs/SFHs_alt_0.5_quenched_all.h5','r')
    ordered_keys = np.sort([int(x.strip('/')) for x in store.keys()])

    #z_array = [float(z) for z in args.z.split(',')]
    z_array = np.arange(args.zlo,args.zhi,args.zstep)
    if args.av_step_type == 'lin':
        av_arr = np.linspace(args.av_lo,args.av_hi,args.n_av)
    elif args.av_step_type == 'log':
        av_arr = np.logspace(args.av_lo,args.av_hi,args.n_av)

    for z in z_array:
        distance_factor = 10.0**(0.4*cosmo.distmod(z).value)
        worker_args = []
        for tf in tqdm(ordered_keys[::-1][np.arange(0,len(ordered_keys),args.time_res)]):   # Iterate through the SFHs for galaxies of different final masses
            sfh_df = store['/'+str(tf)]
            sfh_df = sfh_df[sfh_df['z']>z]
            results = []
            if len(sfh_df)>0:
                worker_args.append([sfh_df,args,av_arr,z,tf,s,bc03_logt_float_array])
        pool_size = 16
        pool = MyPool(processes=pool_size)
        results_df = pd.DataFrame()
        for res_df in tqdm(pool.imap_unordered(sed_worker,worker_args),total=len(worker_args)):
            results_df= results_df.append(res_df)
        pool.close()
        pool.join()
        results_df.to_hdf('/media/data3/wiseman/des/AURA/sims/hostlibs/all_model_params_quench_%s_z%.2f_%.2f_av%.2f_%.2f_rv_rand_full_age_dists_neb_U%.2f_res_%i_beta_%.2f.h5'%(args.templates,args.zlo,args.zhi,av_arr[0],av_arr[-1],args.logU,args.time_res,args.beta),
            key='%.2f'%z)
    print("Done!")
if __name__=="__main__":
    args = parser()
    run(args)
