import glob
import pandas as pd
from tqdm import tqdm
import os
#Do All the combining

#dirname= '/media/data3/wiseman/des/AURA/sims/hostlibs/all_model_params_quench_BC03_z0.200_0.800_av0.00_0.00_rv_rand_full_age_dists_neb_U-2.00_res_10_beta_1.14/'
dirname= '/media/data3/wiseman/des/AURA/sims/hostlibs/20221214'
fn = '/media/data3/wiseman/des/AURA/sims/hostlibs/20221214/all_model_params_quench_bursts_BC03_z0.0005_1.32000_av0.00_1.50_rv_rand_full_age_dists_neb_U-2.00_res_2_beta_1.13_combined.h5'

#full_df = pd.read_hdf(fn,key='main')
full_df = pd.DataFrame()
for fn in tqdm(glob.glob(os.path.join(dirname,'*av0.00_1.50*.h5'))):
    df = pd.read_hdf(fn)
    full_df = full_df.append(df)

for col in full_df.columns:
    try:
        full_df[col]=full_df[col].astype(float)
    except:
        print(col)

#fn = '/media/data3/wiseman/des/AURA/sims/hostlibs/all_model_params_quench_bursts_BC03_z0.14000_1.26000_av0.00_1.50_rv_rand_full_age_dists_neb_U-2.00_res_2_beta_1.14_combined.h5'

fn = '/media/data3/wiseman/des/AURA/sims/hostlibs/20221214/all_model_params_quench_bursts_BC03_z0.0005_1.32000_av0.00_1.50_rv_rand_full_age_dists_neb_U-2.00_res_2_beta_1.13_combined.h5'
full_df.to_hdf(fn,key='main')
for fn in tqdm(glob.glob('/media/data3/wiseman/des/AURA/sims/hostlibs/20221214/SN_ages/*')):
    new_fn = fn.replace('0.dat','0_combined.dat')
    os.rename(fn,new_fn)
