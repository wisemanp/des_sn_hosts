import pystan
import pandas as pd
from des_sn_hosts.utils import stan_utility

def sample_sn_masses(df,model_dir,mass_col='log_m',mass_err_col='logm_err',sfr_col='logssfr', sfr_err_col='logssfr_err',weight_col='weight',index_col = 'CIDint',n_iter=1E4,seed=1234,variable='mass'):
    model_gen = stan_utility.compile_model(model_dir+'generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    if variable=='mass':
        var_col=mass_col
        err_col = mass_err_col
    elif variable=='sfr':
        var_col=sfr_col
        err_col = sfr_err_col
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[var_col].values,
                                  x_err =detections[err_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1,verbose=True)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['zHD','zHDERR',mass_col,mass_err_col,sfr_col,sfr_err_col,weight_col]]
    truthcols.index = truthcols.index.astype(int)
    for col in truthcols.columns:
        df_bootstrapped[col] = truthcols[col]

    #df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped

def sample_field_masses(df,model_dir,mass_col='log_m',mass_err_col='logm_err',sfr_col='log_ssfr',sfr_err_col='logssfr_err',weight_col='weight',
                    index_col = 'CIDint',n_iter=1E4,seed=1234,variable='mass'):

    model_gen = stan_utility.compile_model(model_dir+'generate_mass_sims.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    if variable=='mass':
        var_col=mass_col
        err_col = mass_err_col
    elif variable=='sfr':
        var_col=sfr_col
        err_col = sfr_err_col
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[var_col].values,
                                  x_err =detections[err_col].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1,verbose=True)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['z_spec','z_m2','redshift',#[['SPEC_Z','z_phot','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z',
                                                                mass_col,mass_err_col,sfr_col,sfr_err_col,weight_col]]
    truthcols.index = truthcols.index.astype(int)
    for col in truthcols.columns:
        df_bootstrapped[col] = truthcols[col]
    #df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped

def sample_field_asymm(df,model_dir,mass_col='MASS',mass_err_plus='MASSMAX',mass_err_minus = 'MASSMIN',
                    sfr_col='SPECSFR',sfr_err_plus='SPECSFRMAX',sfr_err_minus='SPECSFRMIN',weight_col='weight',
                    index_col = 'CIDint',n_iter=1E4,seed=1234,variable='mass'):

    model_gen = stan_utility.compile_model(model_dir+'generate_mass_sims_asymm.stan')
    detections = df[df[mass_col]>0]
    nobs=len(detections)
    if variable=='mass':
        var_col=mass_col
        err_col_plus = mass_err_plus
        err_col_minus = mass_err_minus
    elif variable=='sfr':
        var_col=sfr_col
        err_col_plus = sfr_err_plus
        err_col_minus = sfr_err_minus
    fit = model_gen.sampling(data=dict(n_obs=nobs,
                                  x_obs =detections[var_col].values,
                                  x_err_plus =detections[err_col_plus].values - detections[var_col].values+0.001,
                                  x_err_minus = detections[var_col].values - detections[err_col_minus].values+0.001),
                        seed=seed,algorithm='Fixed_param',iter=n_iter,chains=1,verbose=True)
    chain = fit.extract()
    df_bootstrapped = pd.DataFrame(chain['x_sim'].T)
    df_bootstrapped.index = detections[index_col].astype(int)
    truthcols = detections.set_index(index_col,drop=True)[['z_spec','z_m2','redshift',#[['SPEC_Z','z_phot','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z',
                                                                mass_col,mass_err_plus,mass_err_minus,
                                                                sfr_col,sfr_err_plus,sfr_err_minus,
                                                                weight_col]]
    truthcols.index = truthcols.index.astype(int)
    for col in truthcols.columns:
        df_bootstrapped[col] = truthcols[col]
    #df_bootstrapped =df_bootstrapped.merge(truthcols,left_index=True,right_index=True,how='inner')
    return df_bootstrapped

def VVmax(df,z_survey=1,method='ZPEG'):
    from astropy.cosmology import WMAP9 as cosmo

    Vsurvey = cosmo.comoving_volume(z_survey)

    if method=='grid':
        distmod_z = cosmo.distmod(df['redshift'])
        distmod_mag = df['MAG_AUTO_R'] - df['ABSMAGS_R']
        kcorr = distmod_z.value - distmod_mag
        kcorr_z = kcorr/df['redshift']
        from astropy.cosmology import z_at_value
        distmod_max = 26.5 - df['ABSMAGS_R'] +kcorr
        distmod_max = distmod_max.apply(lambda x:x*u.mag)

        zmin = z_at_value(cosmo.distmod, distmod_max.min())
        zmax = z_at_value(cosmo.distmod, distmod_max.max())
        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)
        Dgrid = cosmo.distmod(zgrid)
        zvals = np.interp(distmod_max.apply(lambda x: x.value), Dgrid.value, zgrid)

        vmax = cosmo.comoving_volume(zvals)

        df['VVmax'] = Vsurvey/vmax

        df['VVmax'].clip(lower=1,inplace=True)

    elif method=='ZPEG':

        df['VVmax'] = Vsurvey/cosmo.comoving_volume(df['ZMAX_completeness'])
        df['VVmax'].clip(1,inplace=True)
