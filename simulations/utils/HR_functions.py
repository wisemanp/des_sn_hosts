import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
def add_mass_step(logM,mag=0.1,loc=10):
    return ((logM>loc) * mag*-0.5) + ((logM<loc)* mag*0.5)

def add_age_step(age,mag=0.1,loc=1):
    return ((age>loc) * mag*-0.5) + ((age<loc)* mag*0.5)
def add_age_step_choice(age,mag=0.1):
    return ((age=='old') * mag*-0.5) + ((age=='young')* mag*0.5)

def fit_mass_step(logM,mag=0.1,loc=10):
    return ((logM>loc) * mag*-0.5) + ((logM<loc)* mag*0.5)

def chisq_mu_res_nostep_old(x0,args):
    '''Deprecated. Use chisq_mu_res_nostep'''
    df,params,cosmo = args[0],args[1],args[2]
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

    obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0
    err = df['mB_err']
    return np.sum(((obs-mod)**2)/err**2)


def chisq_mu_res_nostep(x0,args):
    df,params,cosmo = args[0],args[1],args[2]
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

    obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0
    var =  df['mB_err']**2 + alpha**2*df['x1_err']**2 + beta**2*df['c_err']**2
    return np.sum(((obs-mod)**2)/var)

    def chisq_mu_res_nostep_sigint(x0,args):
        df,params,cosmo = args[0],args[1],args[2]
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
        sigint=x0[3]
        mod = cosmo.distmod(df['z']).value

        obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0
        var =  df['mB_err']**2 + alpha**2*df['x1_err']**2 + beta**2*df['c_err']**2
        return np.sum(((obs-mod)**2)/var)

def get_mu_res_nostep(x0,df,params,cosmo):

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
    obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0
    return obs-mod

def chisq_mu_res_step(x0,args):
    df, params,cosmo = args[0], args[1],args[2]
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
    obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0 + step
    err = df['mB_err']
    return np.sum(((obs-mod)**2)/err**2)

def get_mu_res_step(x0,df,params,cosmo):

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
    obs = df['mB'] +alpha*df['x1'] - beta*df['c'] -M0 + step
    return obs-mod

def calculate_step(mu_res,mu_res_err,host_val,split):
    data_left =mu_res[host_val<split]
    data_right =mu_res[host_val>=split]
    errors_left = mu_res_err[host_val<split]
    errors_right = mu_res_err[host_val>=split]

    step = np.average(data_left,weights=1/errors_left**2) - np.average(data_right,weights=1/errors_right**2)
    sig = np.abs(step/np.sqrt((np.std(errors_left)/np.sqrt(len(errors_left)))+(np.std(errors_right)/np.sqrt(len(errors_right)))))
    return step, sig
def get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi):

    if model_c_mids_lo[0] < low['c'][0]:
        pass
    else:
        for x in low.keys():
            print('cutting low')
            low[x] = low[x][1:]


    if model_c_mids_hi[0] < high['c'][0]:
        pass
    else:
        for x in high.keys():
            print('cutting high')
            high[x] = high[x][1:]

    interp_lo = interp1d(np.array(model_c_mids_lo),np.array(model_hr_mids_lo))
    mod_lo = interp_lo(np.array(low['c']))
    interp_hi = interp1d(np.array(model_c_mids_hi),np.array(model_hr_mids_hi))
    mod_hi = interp_hi(np.array(high['c']))

    obs = np.concatenate([np.array(low['hr']),np.array(high['hr'])])
    err = np.concatenate([np.array(low['hr_err']),np.array(high['hr_err'])])
    mod_interp = np.concatenate([mod_lo,mod_hi])
    redchisq = get_red_chisq(obs,mod_interp,err)
    return redchisq

def get_red_chisq_interp2(data_c_mids_lo,data_hr_mids_lo,data_hr_errs_lo,data_c_mids_hi,data_hr_mids_hi,data_hr_errs_hi,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi):

    if model_c_mids_lo[0] < data_c_mids_lo[0]:
        pass
    else:
        data_c_mids_lo = data_c_mids_lo[1:]
        data_hr_mids_lo = data_hr_mids_lo[1:]
        data_hr_errs_lo = data_hr_errs_lo[1:]
    if model_c_mids_hi[0] < data_c_mids_hi[0]:
        pass
    else:

        data_c_mids_hi = data_c_mids_hi[1:]
        data_hr_mids_hi = data_hr_mids_hi[1:]
        data_hr_errs_hi = data_hr_errs_hi[1:]
    interp_lo = interp1d(np.array(model_c_mids_lo),np.array(model_hr_mids_lo))
    mod_lo = interp_lo(np.array(data_c_mids_lo))
    interp_hi = interp1d(np.array(model_c_mids_hi),np.array(model_hr_mids_hi))
    mod_hi = interp_hi(np.array(data_c_mids_hi))

    obs = np.concatenate([np.array(data_hr_mids_lo),np.array(data_hr_mids_hi)])
    err = np.concatenate([np.array(data_hr_errs_lo),np.array(data_hr_errs_hi)])
    mod_interp = np.concatenate([mod_lo,mod_hi])
    redchisq = get_red_chisq(obs,mod_interp,err)
    return redchisq

def get_red_chisq_interp_split_multi(data_c_mids_lo,data_hr_mids_lo,data_hr_errs_lo,data_c_mids_hi,data_hr_mids_hi,data_hr_errs_hi,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi):
    all_obs,all_err,all_mod = [],[],[]
    for split in splits.keys():

        if split['model_c_mids'][0] < split['data_c_mids'][0]:
            pass
        else:
            split['data_c_mids'] = split['data_c_mids'][1:]
            split['data_hr_mids'] = split['data_hr_mids'][1:]
            split['data_hr_errs'] = split['data_hr_errs'][1:]

        interp = interp1d(np.array(split['model_c_mids']),np.array(split['model_hr_mids']))
        mod = interp(split['data_c_mids'])
        all_obs = np.concatenate([all_obs,split['data_hr_mids']])
        all_err = np.concatenate([all_err,split['data_hr_errs']])
        all_mod = np.concatenate([all_mod,mod])
    redchisq = get_red_chisq(all_obs,all_mod,all_err)
    return redchisq

def get_red_chisq_interp_splitx1(obs,model):
    all_obs,all_err,all_mod = [],[],[]
    for key in obs.keys():

        while True:
            if model[key]['c'][0] < obs[key]['c'][0]:
                break
            else:
                for x in obs[key].keys():
                    obs[key][x] = obs[key][x][1:]
        while True:
            if model[key]['c'][-1] > obs[key]['c'][-1]:
                break
            else:
                for x in obs[key].keys():
                    obs[key][x] = obs[key][x][:-1]
        interp = interp1d(np.array(model[key]['c']),np.array(model[key]['hr_mids']))
        mod = interp(np.array(obs[key]['c']))
        all_obs = np.concatenate([all_obs,np.array(obs[key]['hr'])])
        all_err = np.concatenate([all_err,np.array(obs[key]['hr_err'])])
        all_mod = np.concatenate([all_mod,mod])
    redchisq = get_red_chisq(all_obs,all_mod,all_err)
    return redchisq
    
def get_red_chisq_interp_split_multi(splits):
    all_obs,all_err,all_mod = [],[],[]

    for split in splits.keys():
        dat = splits[split]
        while True:
            if dat['model_c_mids'][0] < dat['data_c_mids'][0]:
                break
            else:
                dat['data_c_mids'] = dat['data_c_mids'][1:]
                dat['data_hr_mids'] = dat['data_hr_mids'][1:]
                dat['data_hr_errs'] = dat['data_hr_errs'][1:]
        while True:
            if dat['model_c_mids'][-1] > dat['data_c_mids'][-1]:
                break
            else:
                dat['data_c_mids'] = dat['data_c_mids'][:-1]
                dat['data_hr_mids'] = dat['data_hr_mids'][:-1]
                dat['data_hr_errs'] = dat['data_hr_errs'][:-1]

        interp = interp1d(np.array(dat['model_c_mids']),np.array(dat['model_hr_mids']))
        mod = interp(dat['data_c_mids'])
        all_obs = np.concatenate([all_obs,dat['data_hr_mids']])
        all_err = np.concatenate([all_err,dat['data_hr_errs']])
        all_mod = np.concatenate([all_mod,mod])
    redchisq = get_red_chisq(all_obs,all_mod,all_err)
    return redchisq

def get_red_chisq(obs,mod,err):

    chisq = np.nansum((obs - mod)**2/err**2)
    return chisq/len(err[~np.isnan(err)])
