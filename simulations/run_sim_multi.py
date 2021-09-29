from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter import *
from des_sn_hosts.simulations.utils.plotter_paper import *

Rv_lo_grid = np.arange(1.5,2.5,0.25)
Rv_hi_grid = np.arange(2.4,3.5,0.25)
beta_young_grid = np.arange(2,4.5,0.5)
beta_old_grid = np.arange(2,4.5,0.5)

pth = '/home/wiseman/code/des_sn_hosts/simulations/config/DES_Rv_split_age_2beta_age.yaml'
pth2 = '/media/data3/wiseman/des/AURA/config/DES_Rv_split_age_2beta_age_test.yaml'
from yaml import safe_load as yload
from yaml import safe_dump as ydump
with open(pth,'r') as f:
    c = yload(f)
res = {}
for rv_lo in Rv_lo_grid:
    res[rv_lo] = {}
    for rv_hi in Rv_hi_grid:
        res[rv_lo][rv_hi] = {}
        for beta_young in beta_young_grid:
            for beta_old in beta_old_grid:
            c['SN_rv_model']['params']['rv_young'] = float(rv_hi)
            c['SN_rv_model']['params']['rv_old'] = float(rv_lo)
            c['mB_model']['params']['mu_beta_young']=float(beta_young)
            c['mB_model']['params']['mu_beta_old']=float(beta_old)
            with open(pth2,'w') as f:
                ydump(c,f)
            sim = aura.Sim(pth2)
            n_samples_arr = sim._get_z_dist(des5yr['z'],n=2500)
            zarr=[0.05,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
            sim.sample_SNe(zarr,n_samples_arr,
                           savepath='/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/two_beta/DES_Rv_split_age_2beta_age_test_SN_sim_%.2f_%.2f_%.2f_%.2f.h5'%(rv_lo,rv_hi,beta_old,beta_young))
            sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&\
                                    (sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&(sim.sim_df['c_err']<0.05)]
            sim.sim_df = sim.sim_df[sim.sim_df['mB']<23.35]
            sim.sim_df.to_hdf('/media/data3/wiseman/des/AURA/sims/SNe/for_BBC/two_beta/DES_Rv_split_age_2beta_age_test_SN_sim_%.2f_%.2f_%.2f_%.2f.h5'%(rv_lo,rv_hi,beta_old,beta_young),key='sim')
