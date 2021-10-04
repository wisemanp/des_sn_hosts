from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter import *
from des_sn_hosts.simulations.utils.plotter_paper import *

Rv_lo_grid = np.arange(1.2,2.65,0.1)
Rv_hi_grid = np.arange(2.4,3.5,0.1)
age_step_grid = np.arange(0.0,0.2,0.05)

pth = '/home/wiseman/code/des_sn_hosts/simulations/config/DES_BS20_age_Rv_step_3Gyr_age_x1.yaml'
pth2 = '/media/data3/wiseman/des/AURA/config/DES_BS20_age_Rv_step_3Gyr_age_x1_test.yaml'
from yaml import safe_load as yload
from yaml import safe_dump as ydump
with open(pth,'r') as f:
    c = yload(f)
res = {}
for rv_lo in Rv_lo_grid:
    res[rv_lo] = {}
    for rv_hi in Rv_hi_grid:
        res[rv_lo][rv_hi] = {}
        for age_step in age_step_grid:
            try:
                    c['SN_rv_model']['params']['rv_old'] = float(rv_lo)
                    c['SN_rv_model']['params']['rv_young'] = float(rv_hi)
                    c['mB_model']['params']['age_step']['mag']=float(age_step)
                    with open(pth2,'w') as f:
                        ydump(c,f)
                    sim = aura.Sim(pth2)
                    n_samples_arr = sim._get_z_dist(des5yr['z'],n=5000)
                    zarr=[0.05,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
                    sim.sample_SNe(zarr,n_samples_arr,savepath='/media/data3/wiseman/des/AURA/sims/SNe/DES_2beta_age_Rv_age_test_SN_sim.h5')
                    sim.sim_df = sim.sim_df[(sim.sim_df['x1']<3)&(sim.sim_df['x1']>-3)&(sim.sim_df['c']>-0.3)&(sim.sim_df['c']<0.3)&(sim.sim_df['x1_err']<1)&(sim.sim_df['c_err']<0.05)]
                    sim.sim_df = sim.sim_df[sim.sim_df['mB']<23.35]
                    sim.fit_mu_res()
                    sim.sim_df = sim.sim_df[sim.sim_df['z']>0.1]
                    chis = plot_mu_res_paper(sim)
                    print(rv_lo,rv_hi,age_step)
                    print(chis)
                    chix1,chic = plot_sample_hists(sim)
                    chis.append(chix1)
                    chis.append(chic)
                    res[rv_lo][rv_hi][age_step] = chis
            except:
                    res[rv_lo][rv_hi][age_step]= [-9,-9,-9,-9]
import pickle
with open('/media/data3/wiseman/des/AURA/sims/chi2/chi2_DES_2beta_age_Rv_age_test_small.pkl','wb') as f:
    pickle.dump(res,f)
