from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter_paper import *
from tqdm import tqdm


# read in the BBC output files and calculate the chi-squared to the HR-vs-colour split by mass plot
# read in the BBC output files and calculate the chi-squared to the HR-vs-colour split by mass plot

Rv_lo_grid = np.arange(1.5,2.5,0.25)
Rv_hi_grid = np.arange(2.25,4.0,0.25)
age_step_grid = np.arange(0,0.25,0.05)

pth = '/home/wiseman/code/des_sn_hosts/simulations/config/DES_Rv_linear_age.yaml'
from yaml import safe_load as yload
n=0
chis = np.zeros((len(Rv_lo_grid),len(Rv_hi_grid),len(age_step_grid)),dtype=float)

from tqdm import tqdm
from des_sn_hosts.simulations.utils.plotter_paper import * 
sim = aura.Sim(pth)
                    
with open(pth,'r') as f:
    c = yload(f)
for i, rv_lo in tqdm(enumerate(Rv_lo_grid)):
        for j,rv_hi in tqdm(enumerate(Rv_hi_grid)):
            for k,age_step in enumerate(age_step_grid):
                
                    c['SN_rv_model']['params']['rv_low'] = float(rv_hi)
                    c['SN_rv_model']['params']['rv_high'] = float(rv_lo)
                    c['mB_model']['params']['age_step']['mag'] = float(age_step)
                    sim.config = c
                    
                    from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/age_linear/FITOPT%03d_MUOPT000.FITRES.gz'%n,
                                          delimiter='\s+', comment='#')
                    sim.sim_df = from_bbc
                    sim.sim_df.rename(columns={'U_R':'U-R','MURES':'mu_res','MUERR':'mu_res_err','mBERR':'mB_err'},inplace=True)
                    try:
                        chis[i,j,k] =plot_mu_res_paper_splitssfr(sim)
                    except:
                        chis[i,j,k] =-9999
                    n +=1
np.save('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/age_linear/chis_splitx1.npy',chis)

