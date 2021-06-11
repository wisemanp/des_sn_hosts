import numpy as np
from astropy.cosmology import FlatLambdaCDM

def add_mass_step(logM,mag=0.1,loc=10):
    return ((logM>loc) * mag*-0.5) + ((logM<loc)* mag*0.5)

def add_age_step(age,mag=0.1,loc=1):
    return ((age>loc) * mag*-0.5) + ((age<loc)* mag*0.5)
def add_age_step_choice(age,mag=0.1):
    return ((age=='old') * mag*-0.5) + ((age=='young')* mag*0.5)

def fit_mass_step(logM,mag=0.1,loc=10):
    return ((logM>loc) * mag*-0.5) + ((logM<loc)* mag*0.5)

def chisq_mu_res_nostep(x0,args):
    df,params = args[0],args[1]
    fa,fb=params['fix_alpha'],params['fix_beta']
    if fa=='False':
        alpha=x0[0]
    else:
        alpha=fa
    if fb=='False':
        beta=x0[1]
    else:
        beta=fb
    M0 =x0[2]
    mod = cosmo.distmod(df['z']).value

    obs = df['m_obs'] +alpha*df['x1'] - beta*df['c'] -M0
    err = df['mb_err']
    return np.sum(((obs-mod)**2)/err**2)

def get_mu_res_nostep(x0,df,params):

    fa,fb=params['fix_alpha'],params['fix_beta']
    if fa==False:
        alpha=x0[0]
    else:
        alpha=fa
    if fb==False:
        beta=x0[1]
    else:
        beta=fb
    M0 =x0[2]
    mod = cosmo.distmod(df['z']).value
    obs = df['m_obs'] +alpha*df['x1'] - beta*df['c'] -M0
    return obs-mod

def chisq_mu_res_step(x0,args):
    df, params = args[0], args[1]
    fa,fb=params['fix_alpha'],params['fix_beta']
    if fa==False:
        alpha=x0[0]
    else:
        alpha=fa
    if fb==False:
        beta=x0[1]
    else:
        beta=fb
    gamma=x0[2]
    M0 =x0[3]
    mod = cosmo.distmod(df['z']).value
    step = np.log10(df['mass']).apply(lambda x: fit_mass_step(x,mag=gamma,loc=10))
    obs = df['m_obs'] +alpha*df['x1'] - beta*df['c'] -M0 + step
    err = df['mb_err']
    return np.sum(((obs-mod)**2)/err**2)

def get_mu_res_step(x0,df,params):

    fa,fb=params['fix_alpha'],params['fix_beta']
    if fa==False:
        alpha=x0[0]
    else:
        alpha=fa
    if fb==False:
        beta=x0[1]
    else:
        beta=fb
    gamma=x0[2]
    M0 =x0[3]
    mod = cosmo.distmod(df['z']).value
    step = np.log10(df['mass']).apply(lambda x: fit_mass_step(x,mag=gamma,loc=10))
    obs = df['m_obs'] +alpha*df['x1'] - beta*df['c'] -M0 + step
    return obs-mod

def calculate_step(mu_res,mu_res_err,host_val,split):
    data_left =mu_res[host_val<split]
    data_right =mu_res[host_val>=split]
    errors_left = mu_res_err[host_val<split]
    errors_right = mu_res_err[host_val>=split]

    step = np.average(data_left,weights=1/errors_left**2) - np.average(data_right,weights=1/errors_right**2)
    sig = step/np.sqrt((np.std(errors_left)/np.sqrt(len(errors_left)))+(np.std(errors_right)/np.sqrt(len(errors_right))))
    return step, sig
