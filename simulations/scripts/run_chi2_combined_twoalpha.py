from des_sn_hosts.simulations import aura
from des_sn_hosts.simulations.utils.plotter_paper import *
from tqdm import tqdm
import sys
import matplotlib
matplotlib.rc('figure', max_open_warning = 0)
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
    chi_plots = ['M','UR']
with open(cpath,'r') as f:
    cfg =  yload(f)
alpha_young_grid = np.arange(cfg['alpha_young']['lo'],cfg['alpha_young']['hi'],cfg['alpha_young']['step'])
alpha_old_grid = np.arange(cfg['alpha_old']['lo'],cfg['alpha_old']['hi'],cfg['alpha_old']['step'])
age_step_grid = np.arange(cfg['age_step']['lo'],cfg['age_step']['hi'],cfg['age_step']['step'])
pth = cfg['config_path']
model_config = os.path.split(pth)[-1]
model_name = model_config.split('.')[0]

n=0
chis = np.zeros((len(Rv_lo_grid),len(Rv_hi_grid),len(mass_step_grid)),dtype=float)

from tqdm import tqdm
from des_sn_hosts.simulations.utils.plotter_paper import *
sim = aura.Sim(pth)

with open(pth,'r') as f:
    c = yload(f)
for i,alpha_young in tqdm(enumerate(alpha_young_grid)):
    for j,alpha_old in tqdm(enumerate(alpha_old_grid)):
        for k,age_step in tqdm(enumerate(age_step_grid)):
                    c['mB_model']['params']['mu_alpha_young'] = float(alpha_young)
                    c['mB_model']['params']['mu_alpha_old'] = float(alpha_old)
                    c['mB_model']['params']['age_step']['mag'] = float(age_step)
                    sim.config = c
                    if BBC =='5D':
                        from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/BBC5D/FITOPT%03d_MUOPT000.FITRES.gz'%(cfg['save']['dir'],n),
                                          delimiter='\s+', comment='#')
                    elif BBC =='0D':
                        from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/0D/FITOPT%03d_MUOPT000.FITRES.gz'%(cfg['save']['dir'],n),
                                          delimiter='\s+', comment='#')
                    else:

                        from_bbc = pd.read_csv('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/FITOPT%03d_MUOPT000.FITRES.gz'%(cfg['save']['dir'],n),
                                          delimiter='\s+', comment='#')
                    sim.sim_df = from_bbc
                    sim.sim_df.rename(columns={'U_R':'U-R','MURES':'mu_res','MUERR':'mu_res_err','mBERR':'mB_err'},inplace=True)
                    try:
                        if BBC=='5D':
                            chis[i,j,k] =np.sum(plot_mu_res_paper_combined_new(sim,y5data='5D',chi_plots = chi_plots))
                        elif BBC=='0D':
                            chis[i,j,k] =np.sum(plot_mu_res_paper_combined_new(sim,y5data='0D',chi_plots = chi_plots))
                        else:
                            chis[i,j,k] =np.sum(plot_mu_res_paper_combined_new(sim,chi_plots = chi_plots))
                    except:
                        chis[i,j,k] =-9999
                    n +=1
np.save('/media/data3/wiseman/des/AURA/sims/SNe/from_BBC/%s/chis_combined_BBC%s.npy'%(cfg['save']['dir'],BBC),chis)
