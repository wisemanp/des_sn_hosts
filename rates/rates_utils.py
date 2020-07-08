import pystan
import pandas as pd
from des_sn_hosts.utils import stan_utility

def sample_sn_masses(df,model_dir,mass_col='log_m',mass_err_col='logm_err',sfr_col='log_ssfr', sfr_err_col='log_ssfr_err',index_col = 'CIDint',n_iter=1E4,seed=1234,):

    model_gen = stan_utility.compile_model(model_dir+'generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[mass_col].values,
                                  x_err =detections[mass_err_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1,verbose=True)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['zHD','zHDERR',mass_col,mass_err_col,sfr_col,sfr_err_col]]
    truthcols.index = truthcols.index.astype(int)
    for col in truthcols.columns:
        df_bootstrapped[col] = truthcols[col]

    #df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped

def sample_field_masses(df,model_dir,mass_col='log_m',mass_err_col='logm_err',sfr_col='log_ssfr',sfr_err_col='logssfr_err',
                    index_col = 'CIDint',n_iter=1E4,seed=1234):

    model_gen = stan_utility.compile_model(model_dir+'generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[mass_col].values,
                                  x_err =detections[mass_err_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1,verbose=True)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['z_spec','z_phot',#[['SPEC_Z','z_phot','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z',
                                                                mass_col,mass_err_col,sfr_col,sfr_err_col]]
    truthcols.index = truthcols.index.astype(int)
    for col in truthcols.columns:
        df_bootstrapped[col] = truthcols[col]
    #df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped
