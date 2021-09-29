from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter import *
from des_sn_hosts.simulations.utils.plotter_paper import *
from des_sn_hosts.utils.utils import MyPool
from tqdm import tqdm

Rv_lo_grid = np.arange(1.5,2.5,0.25)
Rv_hi_grid = np.arange(2.25,4.0,0.25)
age_step_grid = np.arange(0,0.25,0.05)
pth = '/home/wiseman/code/des_sn_hosts/simulations/config/DES_Rv_linear_age.yaml'
pth2 = '/media/data3/wiseman/des/AURA/config/DES_Rv_linear_age_test.yaml'


from yaml import safe_load as yload
from yaml import safe_dump as ydump

import multiprocessing
def sim_worker(args):
    sim = aura.Sim(pth)
    rv_hi,rv_lo,age_step = [args[i] for i in range(3)]
    with open(pth,'r') as f:
        c = yload(f)
    c['SN_rv_model']['params']['rv_low'] = float(rv_hi)
    c['SN_rv_model']['params']['rv_high'] = float(rv_lo)
    c['mB_model']['params']['age_step']['mag'] = float(age_step)
    sim.config = c
    n_samples_arr = sim._get_z_dist(des5yr['z'],n=2500)
    zarr=[0.05,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
    sim.sample_SNe(zarr,n_samples_arr,
                   savepath='/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/age_linear/DES_Rv_linear_age_test_SN_sim_%.2f_%.2f_%.2f.h5'%(rv_lo,rv_hi,age_step))
    sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&\
                            (sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&(sim.sim_df['c_err']<0.05)]
    sim.sim_df = sim.sim_df[sim.sim_df['mB']<23.35]
    sim.sim_df.to_hdf('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/age_linear/DES_Rv_linear_age_test_SN_sim_%.2f_%.2f_%.2f.h5'%(rv_lo,rv_hi,age_step),key='sim')

def multi_sim(args):
    
    pool_size = 8
    pool = MyPool(processes=pool_size)
    for _ in tqdm(pool.imap_unordered(sim_worker,args),total=len(args)):
        pass
    pool.close()
    pool.join()
    
    

if __name__=='__main__':   
    args = []
    for rv_lo in Rv_lo_grid:
        for rv_hi in Rv_hi_grid:
            for age_step in age_step_grid:
                args.append([rv_hi,rv_lo,age_step])
    multi_sim(args)
