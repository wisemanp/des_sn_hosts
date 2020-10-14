import pystan
import pandas as pd
import numpy as np
import progressbar
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

def sample_field_masses(df,model_dir,mass_col='log_m',mass_err_col='logm_err',sfr_col='log_ssfr',sfr_err_col='logssfr_err',weight_col='weight',index_col = 'CIDint',n_iter=1E4,seed=1234,variable='mass'):

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

    return df

def split_by_z(df,fn,zcol='zHD',zmin=0.2,zmax=1.2,zstep=0.2,do_VVmax=False):
    groups = df.groupby(pd.cut(df[zcol],bins=np.linspace(zmin,zmax,int((zmax-zmin)/zstep)+1,endpoint=True)))
    for n,g in groups:
        if do_VVmax:
            g = VVmax(g,z_survey=n.right)
        g.to_hdf(fn,key='z_%.2f_%.2f'%(n.left,n.right))


def SN_G_MC(sn_samples_mass,field_samples_mass,n_samples=1E4,mmin=7.25,mmax=13,mstep=0.25,savename=None,variable='mass',key_ext=None, weight_col_SN='weight',weight_col_field='weight',rate_corr=-0.38):
    mbins = np.linspace(mmin,mmax,((mmax-mmin)/mstep)+1)
    iter_df = pd.DataFrame(columns = range(0,int(n_samples),1),index=mbins+0.125)

    with progressbar.ProgressBar(max_value = n_samples) as bar:
        for i in range(0,n_samples):
            snmassgroups =sn_samples_mass.groupby(pd.cut(sn_samples_mass[i],
                                                 bins=mbins))[[i,weight_col_SN]]
            i_f = np.random.randint(0,100)
            fieldmassgroups = field_samples_mass.groupby( pd.cut(field_samples_mass[i_f],
                                                        bins=mbins))[[i_f,weight_col_field]]
            xs = []
            ys = []

            for (n,g),(n2,g2) in zip(snmassgroups,fieldmassgroups):

                if g.size >0 and g2.size>0:
                    xs.append(n.mid)

                    ys.append(np.log10(g[weight_col_SN].sum()/g2[weight_col_field].sum())+rate_corr) # We want a per-year rate.

            xs = np.array(xs)
            ys = np.array(ys)
            entry = pd.Series(ys,index=xs)
            iter_df.loc[entry.index,i] = entry
            bar.update(i)
    if key_ext:
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_%s_%s'%(variable,key_ext))
    else:
        iter_df.to_hdf(savename,index=True,key='bootstrap_samples_%s'%variable)
    return iter_df
