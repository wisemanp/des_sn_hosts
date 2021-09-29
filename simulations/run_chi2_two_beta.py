from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter_paper import *
from tqdm import tqdm


# read in the BBC output files and calculate the chi-squared to the HR-vs-colour split by mass plot
Rv_lo_grid = np.arange(1.5,2.,0.25)
Rv_hi_grid = np.arange(2.75,3.5,0.25)
beta_young_grid = np.arange(1.6,2.6,0.2)
beta_old_grid = np.arange(1.6,2.6,0.2)
pth = '/home/wiseman/code/des_sn_hosts/simulations/config/DES_Rv_split_age_2beta_age.yaml'
from yaml import safe_load as yload
n=0
chis = np.zeros((len(Rv_lo_grid),len(Rv_hi_grid),len(beta_young_grid),len(beta_old_grid)),dtype=float)

from tqdm import tqdm
from des_sn_hosts.simulations.utils.plotter_paper import * 
sim = aura.Sim(pth)
                    
with open(pth,'r') as f:
    c = yload(f)
for i, rv_lo in tqdm(enumerate(Rv_lo_grid)):
        for j,rv_hi in tqdm(enumerate(Rv_hi_grid)):
            for k,beta_young in enumerate(beta_young_grid):
                for l,beta_old in enumerate(beta_old_grid):
                    print('Rv young',rv_hi,'Rv old: ',rv_lo,'Beta young: ',beta_young,'beta old: ',beta_old)
                    c['SN_rv_model']['params']['rv_young'] = float(rv_hi)
                    c['SN_rv_model']['params']['rv_old'] = float(rv_lo)
                    c['mB_model']['params']['mu_beta_young']=float(beta_young)
                    c['mB_model']['params']['mu_beta_old']=float(beta_old)
                    sim.config = c
                    
                    from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/two_beta/small_grid/FITOPT%03d_MUOPT000.FITRES.gz'%n,
                                          delimiter='\s+', comment='#')
                    sim.sim_df = from_bbc
                    sim.sim_df.rename(columns={'U_R':'U-R','MURES':'mu_res','MUERR':'mu_res_err','mBERR':'mB_err'},inplace=True)
                    try:
                        chis[i,j,k,l] =np.sum(plot_mu_res_paper(sim))
                    except:
                        chis[i,j,k,l] =-9999
                    n+=1
np.save('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/two_beta/small_grid/chis.npy',chis)

