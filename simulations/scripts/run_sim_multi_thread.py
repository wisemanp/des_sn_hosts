from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter import *
from des_sn_hosts.simulations.utils.plotter_paper import *
from des_sn_hosts.utils.utils import MyPool
from tqdm import tqdm
import sys
from yaml import safe_load as yload
from yaml import safe_dump as ydump
import csv
import numpy as np

import multiprocessing
des5yr = pd.read_hdf(os.path.join(aura_dir,'data','DES5YR_MV20200701_Hosts20211018_BBC1D.h5'))

des5yr = des5yr[des5yr['SPECZ']<0.7]
def load_config(cpath):
    with open(cpath,'r') as f:
        return yload(f)

def prep_df_for_BBC(df):
    df = df[df['mB']<25]
    df = df[(df['x1']<3)&(df['x1']>-3)&(df['c']>-0.3)&(df['c']<0.3)&\
                           (df['x1_err']<1)\
                           &(df['c_err']<0.1)    # uncomment to include a colour error cut
                           ]
    df['CID'] = np.arange(len(df),dtype=int)
    #df['CID'] = df['CID'].astype(int)
    df['IDSURVEY'] = 10
    df['TYPE'] = 101
    df.rename(columns={'z':'zHD','mB_err':'mBERR','x1_err':'x1ERR','c_err':'cERR',
                             },inplace=True)
    df = df[df['zHD']>=0.15]
    df['zHDERR'] = 0.0001
    df['zCMB'] = df['zHD']
    df['zCMBERR'] = 0.0001
    df['zHEL'] = df['zHD']
    df['zHELERR'] = 0.0001
    df['VPEC'] =0
    df['VPECERR'] =0
    df['x0']=10**(-0.4*(df['mB']-10.6350))
    df['x0ERR']=0.4*np.log(10)*df['x0'].values*df['mBERR']
    df['COV_x1_x0'] =0
    df['COV_c_x0'] =0
    df['COV_x1_c'] =0
    df['PKMJD'] =56600
    df['PKMJDERR'] =0.1
    df['FITPROB'] =1
    df['PROB_SNNTRAINV19_z_TRAINDES_V19']=1
    df['HOST_LOGMASS'] = np.log10(df['mass'])
    df['HOST_LOG_SFR'] = np.log10(df['sfr'])
    df['HOST_LOG_sSFR'] = np.log10(df['ssfr'])
    df['VARNAMES:'] = 'SN:'
    columns=['VARNAMES:','CID', 'IDSURVEY', 'TYPE', 'mB', 'mBERR', 'cERR', 'x1ERR', 'zHD', 'TYPE', 'zHDERR', 'zCMB',
           'zCMBERR', 'zHEL', 'zHELERR','x0', 'x0ERR','COV_x1_x0', 'COV_c_x0', 'COV_x1_c',
             'VPEC', 'VPECERR', 'PKMJD', 'PKMJDERR', 'FITPROB',
           'PROB_SNNTRAINV19_z_TRAINDES_V19', 'HOST_LOGMASS', 'HOST_LOG_SFR',
           'HOST_LOG_sSFR', 'distmod', 'mass', 'ssfr', 'sfr', 'mean_ages',
           'SN_age', 'rv', 'E', 'host_Av', 'U-R', 'c', 'c_noise', 'c_int', 'x1','x1_int','x1_noise'
           #'prog_age'
            ]
    return df,columns

def sim_worker(args):

    rv_hi,rv_lo,age_step,cfg = [args[i] for i in range(4)]
    pth = cfg['config_path']
    model_config = os.path.split(pth)[-1]
    model_name = model_config.split('.')[0]
    sim = aura.Sim(pth)
    with open(pth,'r') as f:
        c = yload(f)
    if c['SN_rv_model']['model']=='age_rv_step':
        c['SN_rv_model']['params']['rv_young'] = float(rv_hi)
        c['SN_rv_model']['params']['rv_old'] = float(rv_lo)

    else:
        c['SN_rv_model']['params']['rv_low'] = float(rv_hi)
        c['SN_rv_model']['params']['rv_high'] = float(rv_lo)
    c['mB_model']['params']['age_step']['mag'] = float(age_step)
    sim.config = c
    #n_samples_arr = sim._get_z_dist(des5yr['zHD'],n=cfg['n_samples'])
    zs = np.linspace(0,1,100)
    zs_cubed = zs**2.5
    numbers = np.random.choice(zs,p=zs_cubed/np.sum(zs_cubed),size=cfg['n_samples'])
    zarr = np.arange(0.14,0.84,0.04)
    n_samples_arr = sim._get_z_dist(numbers,n=cfg['n_samples'],frac_low_z=0.,zbins=zarr+0.02)

    if not os.path.isdir(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir'])):
        os.mkdir(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir']))
    if not os.path.isdir(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/',cfg['save']['dir'])):
        os.mkdir(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/',cfg['save']['dir']))
    sim.sample_SNe(zarr,n_samples_arr,
                   savepath=os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir'],
                   '%s_test_SN_sim_%.2f_%.2f_%.2f.h5'%(model_name,rv_lo,rv_hi,age_step)))
    sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&\
                            (sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&\
                            (sim.sim_df['c_err']<0.1)   # uncomment to include a colour error cut
                            ]
    sim.sim_df = sim.sim_df[sim.sim_df['mB']<25]
    sim.sim_df = sim.sim_df[sim.sim_df['eff_mask']==1]
    sim.sim_df = sim.sim_df[sim.sim_df['z']<=0.7]
    sim.sim_df.to_hdf(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir'],
        '%s_test_SN_sim_%.2f_%.2f_%.2f.h5'%(model_name,rv_lo,rv_hi,age_step)),key='sim')

def multi_sim(args):

    pool_size = 8
    pool = MyPool(processes=pool_size)
    for _ in tqdm(pool.imap_unordered(sim_worker,args),total=len(args)):
        pass
    pool.close()
    pool.join()



if __name__=='__main__':
    cpath = sys.argv[1]
    cfg = load_config(cpath)
    Rv_lo_grid = np.arange(cfg['Rv_lo']['lo'],cfg['Rv_lo']['hi'],cfg['Rv_lo']['step'])
    Rv_hi_grid = np.arange(cfg['Rv_hi']['lo'],cfg['Rv_hi']['hi'],cfg['Rv_hi']['step'])
    age_step_grid = np.arange(cfg['age_step']['lo'],cfg['age_step']['hi'],cfg['age_step']['step'])

    args = []
    for rv_lo in Rv_lo_grid:
        for rv_hi in Rv_hi_grid:
            for age_step in age_step_grid:
                args.append([rv_hi,rv_lo,age_step,cfg])
    multi_sim(args)
    print('Simulations complete. Now converting to BBC format')
    n=0
    param_map = {}
    pth = cfg['config_path']
    model_config = os.path.split(pth)[-1]
    model_name = model_config.split('.')[0]
    for rv_lo in Rv_lo_grid:
        for rv_hi in Rv_hi_grid:
            for age_step in age_step_grid:
                df = pd.read_hdf(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir'],
                    '%s_test_SN_sim_%.2f_%.2f_%.2f.h5'%(model_name,rv_lo,rv_hi,age_step)))
                df,cols = prep_df_for_BBC(df)
                df[cols].to_csv(os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',
                        cfg['save']['dir'],'FITOPT%03d.FITRES'%n),
                    index=False, sep=' ', quoting=csv.QUOTE_NONE, quotechar="",escapechar=" ")
                param_map[n] = [rv_lo,rv_hi,age_step]
                n+=1
    pickle.dump(param_map,
        open('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/%s/param_name_map.pkl'%(cfg['save']['dir']),'wb'))
