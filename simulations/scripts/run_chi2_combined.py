from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter_paper import *
from tqdm import tqdm
import sys

# read in the BBC output files and calculate the chi-squared to the HR-vs-colour split by mass plot
# read in the BBC output files and calculate the chi-squared to the HR-vs-colour split by mass plot


from yaml import safe_load as yload
cpath = sys.argv[1]
try:
    BBC = sys.argv[2]
except:
    BBC = '1D'
try:
    chi_plots = sys.argv[3].split(',')
except:
    chi_plots = ['M','UR','sSFR']
with open(cpath,'r') as f:
    cfg =  yload(f)
Rv_lo_grid = np.arange(cfg['Rv_lo']['lo'],cfg['Rv_lo']['hi'],cfg['Rv_lo']['step'])
Rv_hi_grid = np.arange(cfg['Rv_hi']['lo'],cfg['Rv_hi']['hi'],cfg['Rv_hi']['step'])
age_step_grid = np.arange(cfg['age_step']['lo'],cfg['age_step']['hi'],cfg['age_step']['step'])
pth = cfg['config_path']
model_config = os.path.split(pth)[-1]
model_name = model_config.split('.')[0]

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
                    if BBC =='5D':
                        from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/BBC5D/FITOPT%03d_MUOPT000.FITRES.gz'%(cfg['save']['dir'],n),
                                          delimiter='\s+', comment='#')
                    else:

                        from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/FITOPT%03d_MUOPT000.FITRES.gz'%(cfg['save']['dir'],n),
                                          delimiter='\s+', comment='#')
                    sim.sim_df = from_bbc
                    sim.sim_df.rename(columns={'U_R':'U-R','MURES':'mu_res','MUERR':'mu_res_err','mBERR':'mB_err'},inplace=True)
                    try:
                        if BBC=='5D':
                            chis[i,j,k] =np.sum(plot_mu_res_paper_combined_new(sim,y5data='5D',chi_plots = chi_plots))
                        else:
                            chis[i,j,k] =np.sum(plot_mu_res_paper_combined_new(sim,chi_plots = chi_plots))
                    except:
                        chis[i,j,k] =-9999
                    n +=1
np.save('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/chis_combined_BBC%s.npy'%(cfg['save']['dir'],BBC),chis)
