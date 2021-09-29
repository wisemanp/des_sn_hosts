from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter import *
from des_sn_hosts.simulations.utils.plotter_paper import *
from des_sn_hosts.utils.utils import MyPool
from tqdm import tqdm


from yaml import safe_load as yload
from yaml import safe_dump as ydump

import multiprocessing

def load_config(cpath):
    with open(cpath,'r') as f:
        return yload(f)

def sim_worker(args):
    sim = aura.Sim(pth)
    rv_hi,rv_lo,age_step,cfg = [args[i] for i in range(4)]
    with open(pth,'r') as f:
        c = yload(f)
    c['SN_rv_model']['params']['rv_low'] = float(rv_hi)
    c['SN_rv_model']['params']['rv_high'] = float(rv_lo)
    c['mB_model']['params']['age_step']['mag'] = float(age_step)
    sim.config = c
    n_samples_arr = sim._get_z_dist(des5yr['z'],n=2500)
    zarr=[0.05,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
    pth = cfg['config_path']
    model_config = os.path.split(pth)[-1]
    model_name = model_config.split('.')[0]
    sim.sample_SNe(zarr,n_samples_arr,
                   savepath=os.path.join('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/',cfg['save']['dir'],
                   '%s_test_SN_sim_%.2f_%.2f_%.2f.h5'%(model_name,rv_lo,rv_hi,age_step)))
    sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&\
                            (sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&(sim.sim_df['c_err']<0.05)]
    sim.sim_df = sim.sim_df[sim.sim_df['mB']<23.35]
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
