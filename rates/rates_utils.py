import pystan
import pandas as pd
from des_sn_hosts.utils import stan_utility

def sample_sn_masses(df,mass_col='log_m',err_col='logm_err',index_col = 'CIDint',n_iter=1E4,seed=1234):

    model_gen = stan_utility.compile_model(r.root_dir+'models/generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[mass_col].values,
                                  x_err =detections[mass_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['zHD','zHDERR','HOST_LOGMASS','HOST_LOGMASS_ERR','VVmax']]
    truthcols.index = truthcols.index.astype(int)
    df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped

def sample_field_masses(df,mass_col='log_m',err_col='logm_err',index_col = 'CIDint',n_iter=1E4,seed=1234):

    model_gen = stan_utility.compile_model(r.root_dir+'models/generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[mass_col].values,
                                  x_err =detections[mass_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['SPEC_Z','z_phot','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z',
                                                                mass_col,err_col]]
    truthcols.index = truthcols.index.astype(int)
    df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped
