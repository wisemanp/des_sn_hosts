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
        mod = interp(split['data_c_mids']))
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
    
def plot_mu_res_paper_combined_new(sim,obs=True,label_ext='',colour_split=1,mass_split=1E+10,return_chi=True,data='new'):
    data_mass_split = np.log10(mass_split)
    chis = []
    fMASSUR,(axMASS,axUR,axsSFR)=plt.subplots(1,3,figsize=(16,6.6),sharey=True)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi =[],[],[],[],[],[]
    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['mass']>mass_split]


            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['mass']<=mass_split]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass

    axMASS.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Low Mass')
    axMASS.fill_between(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)

    axMASS.plot(model_c_mids_hi ,model_hr_mids_hi,c=split_colour_2,lw=3,label='Model High Mass',ls='--')
    axMASS.fill_between(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),np.array(model_hr_mids_hi)+np.array(model_hr_errs_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)
    #axMASS.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    if obs and data=='old':
        low =lisa_data['global_mass']['low']
        for x in low.keys():
            if x!='c':
                low[x] = np.array(low[x])[~np.isnan(low['c'])]
        low['c']=np.array(low['c'])[~np.isnan(low['c'])]
        high=lisa_data['global_mass']['high']
        for x in high.keys():
            if x!='c':
                high[x] = np.array(high[x])[~np.isnan(high['c'])]
        high['c'] = np.array(high['c'])[~np.isnan(high['c'])]
        axMASS.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr=low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})<10$')
        axMASS.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr=high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})>10$')
        chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        axMASS.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)

    if obs and data=='new':
        data_c_mids_lo , data_hr_mids_lo , data_hr_errs_lo , data_c_mids_hi , data_hr_mids_hi ,  data_hr_errs_hi =[],[],[],[],[],[]
        for counter,(n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['c'],bins=np.linspace(-0.3,0.3,10)))):

            g1 = g[g['massmc']>data_mass_split]

            g1 = g1.dropna(subset=['MURES'])

            if len(g1)>0:

                data_hr_mids_hi.append(np.average(g1['MURES'],weights=1/g1['MUERR']**2))
                data_hr_errs_hi.append(g1['MURES'].std()/np.sqrt(len(g1['MURES'])))
                data_c_mids_hi.append(n.mid)


            g2 = g[g['massmc']<=data_mass_split]

            g2 = g2.dropna(subset=['MURES'])

            if len(g2)>0:

                data_hr_mids_lo.append(np.average(g2['MURES'],weights=1/g2['MUERR']**2))
                data_hr_errs_lo.append(g2['MURES'].std()/np.sqrt(len(g2['MURES'])))
                data_c_mids_lo.append(n.mid)

        axMASS.errorbar(data_c_mids_lo,data_hr_mids_lo,yerr=data_hr_errs_lo,marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})<10$')
        axMASS.errorbar(data_c_mids_hi,data_hr_mids_hi,yerr=data_hr_errs_hi,marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})>10$')
        splits = {'lo':{'data_c_mids':data_c_mids_lo,'data_hr_mids':data_hr_mids_lo,'data_hr_errs':data_hr_errs_lo,'model_c_mids':model_c_mids_lo,'model_hr_mids':model_hr_mids_lo},
           'hi':{'data_c_mids':data_c_mids_hi,'data_hr_mids':data_hr_mids_hi,'data_hr_errs':data_hr_errs_hi,'model_c_mids':model_c_mids_hi,'model_hr_mids':model_hr_mids_hi}
              }

        chisq = get_red_chisq_interp_split_multi(splits)
        axMASS.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)

    axMASS.set_xlabel('$c$',size=24)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=24,)
    axMASS.legend(fontsize=13,loc='lower center')
    #axMASS.set_title(sim.save_string + '_paper',size=20)
    axMASS.set_ylim(-0.2,0.2)
    axMASS.set_xlim(-0.19,0.3)
    axMASS.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axMASS.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    axMASS.tick_params(which='both',right=True,top=True,labelsize=16)
    #plt.savefig(sim.fig_dir +'HR_vs_c_split_mass_%s'%(sim.save_string + '_paper')+label_ext)
    #fUR,axUR=plt.subplots(figsize=(8,6.5))
    #axUR.set_title(sim.save_string + '_paper',size=20)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi =[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['U-R']>colour_split]

            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['U-R']<=colour_split]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass

    axUR.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Blue Host')
    axUR.fill_between(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)
    axUR.plot(model_c_mids_hi ,model_hr_mids_hi,c=split_colour_2,lw=3,label='Model Red Host',ls='--')
    axUR.fill_between(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),np.array(model_hr_mids_hi)+np.array(model_hr_errs_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)

    if obs and data =='old':
        low =lisa_data['global_U-R']['low']
        for x in low.keys():
            if x!='c':
                low[x] = np.array(low[x])[~np.isnan(low['c'])]
        low['c']=np.array(low['c'])[~np.isnan(low['c'])]
        high=lisa_data['global_U-R']['high']
        for x in high.keys():
            if x!='c':
                high[x] = np.array(high[x])[~np.isnan(high['c'])]
        high['c'] = np.array(high['c'])[~np.isnan(high['c'])]
        chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        axUR.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr =low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
        axUR.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr =high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
        axUR.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)

    if obs and data=='new':
        data_c_mids_lo , data_hr_mids_lo , data_hr_errs_lo , data_c_mids_hi , data_hr_mids_hi ,  data_hr_errs_hi =[],[],[],[],[],[]
        for counter,(n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['c'],bins=np.linspace(-0.3,0.3,10)))):

            g1 = g[g['U-R']>colour_split]

            g1 = g1.dropna(subset=['MURES'])

            if len(g1)>0:

                data_hr_mids_hi.append(np.average(g1['MURES'],weights=1/g1['MUERR']**2))
                data_hr_errs_hi.append(g1['MURES'].std()/np.sqrt(len(g1['MURES'])))
                data_c_mids_hi.append(n.mid)


            g2 = g[g['U-R']<=colour_split]

            g2 = g2.dropna(subset=['MURES'])

            if len(g2)>0:

                data_hr_mids_lo.append(np.average(g2['MURES'],weights=1/g2['MUERR']**2))
                data_hr_errs_lo.append(g2['MURES'].std()/np.sqrt(len(g2['MURES'])))
                data_c_mids_lo.append(n.mid)

        axUR.errorbar(data_c_mids_lo,data_hr_mids_lo,yerr=data_hr_errs_lo,marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
        axUR.errorbar(data_c_mids_hi,data_hr_mids_hi,yerr=data_hr_errs_hi,marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
        splits = {'lo':{'data_c_mids':data_c_mids_lo,'data_hr_mids':data_hr_mids_lo,'data_hr_errs':data_hr_errs_lo,'model_c_mids':model_c_mids_lo,'model_hr_mids':model_hr_mids_lo},
           'hi':{'data_c_mids':data_c_mids_hi,'data_hr_mids':data_hr_mids_hi,'data_hr_errs':data_hr_errs_hi,'model_c_mids':model_c_mids_hi,'model_hr_mids':model_hr_mids_hi}
              }

        chisq = get_red_chisq_interp_split_multi(splits)
        axUR.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)
    axUR.set_xlabel('$c$',size=24)
    #axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axUR.legend(fontsize=13,loc='lower left')
    axUR.set_ylim(-0.2,0.2)
    axUR.set_xlim(-0.19,0.3)
    plt.savefig('test')
    l= axUR.get_xticklabels()
    print(l[0],l[1],l[2])
    l[0]=matplotlib.text.Text(-0.2,0,' ')
    axUR.set_xticklabels(l)
    axUR.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axUR.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    axUR.tick_params(right=True,top=True,which='both',labelsize=16)
    mvdf = pd.read_csv('/media/data3/wiseman/des/AURA/data/data_nozcut_snnv19.csv')

    oldest,oldish,youngish,youngest = ['#FF9100','red','purple','darkblue']


    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo ,model_c_mids_mid , model_hr_mids_mid , model_hr_errs_mid = [],[],[],[],[],[]
    model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi = [],[],[]
    x1_split=-0.3
    ssfr_lo_split = -11
    sim_ssfr_lo_split = ssfr_lo_split+0.5
    ssfr_hi_split = -9.5
    sim_ssfr_hi_split = ssfr_hi_split+0.5
    sim.sim_df['log_ssfr'] = np.log10(sim.sim_df['ssfr'])
    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):

        g1 = g[g['log_ssfr']>sim_ssfr_hi_split]

        if len(g1)>0:
            model_c_mids_hi.append(n.mid)
            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))

        g2 = g[(g['log_ssfr']<=sim_ssfr_hi_split)&(g['log_ssfr']>sim_ssfr_lo_split)]

        if len(g2)>0:

            model_c_mids_mid.append(n.mid)
            model_hr_mids_mid.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_mid.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))

        g3 = g[g['log_ssfr']<=sim_ssfr_lo_split]

        if len(g3)>0:
            model_c_mids_lo.append(n.mid)
            model_hr_mids_lo.append(np.average(g3['mu_res'],weights=1/g3['mu_res_err']**2))
            model_hr_errs_lo.append(g3['mu_res'].std()/np.sqrt(len(g3['mu_res'])))



    axsSFR.plot(model_c_mids_lo ,model_hr_mids_lo,lw=1,label='Model $\log(\mathrm{sSFR}/\mathrm{yr}^{-1})<11$',color=oldest)
    axsSFR.fill_between(model_c_mids_lo,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),color=oldest,lw=0.5,ls=':',alpha=0.05)

    axsSFR.plot(model_c_mids_mid ,model_hr_mids_mid,lw=1,label='Model $-11\leq\log(\mathrm{sSFR}/\mathrm{yr}^{-1})<9.5$',ls=':',color=oldish)
    axsSFR.fill_between(model_c_mids_mid ,np.array(model_hr_mids_mid)-np.array(model_hr_errs_mid),np.array(model_hr_mids_mid)+np.array(model_hr_errs_mid),color=oldish,lw=0.5,ls=':',alpha=0.05)


    axsSFR.plot(model_c_mids_hi,model_hr_mids_hi,lw=1,label='Model $\log(\mathrm{sSFR}/\mathrm{yr}^{-1})>-9.5$',ls='--',color=youngest)
    axsSFR.fill_between(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),np.array(model_hr_mids_hi)+np.array(model_hr_errs_hi),color=youngest,lw=0.5,ls=':',alpha=0.05)
    if obs and data=='new':
        data_c_mids_lo , data_hr_mids_lo , data_hr_errs_lo , data_c_mids_mid , data_hr_mids_mid ,  data_hr_errs_mid, data_c_mids_hi , data_hr_mids_hi ,  data_hr_errs_hi =[],[],[],[],[],[],[],[],[]
        for counter,(n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['c'],bins=np.linspace(-0.3,0.3,10)))):

            g1 = g[g['ssfr']<ssfr_lo_split]

            g1 = g1.dropna(subset=['MURES'])

            if len(g1)>0:

                data_hr_mids_lo.append(np.average(g1['MURES'],weights=1/g1['MUERR']**2))
                data_hr_errs_lo.append(g1['MURES'].std()/np.sqrt(len(g1['MURES'])))
                data_c_mids_lo.append(n.mid)


            g2 = g[(g['ssfr']>=ssfr_lo_split)&(g['ssfr']<ssfr_hi_split)]

            g2 = g2.dropna(subset=['MURES'])

            if len(g2)>0:

                data_hr_mids_mid.append(np.average(g2['MURES'],weights=1/g2['MUERR']**2))
                data_hr_errs_mid.append(g2['MURES'].std()/np.sqrt(len(g2['MURES'])))
                data_c_mids_mid.append(n.mid)

            g3 = g[g['ssfr']>=ssfr_hi_split]

            g3 = g3.dropna(subset=['MURES'])

            if len(g3)>0:

                data_hr_mids_hi.append(np.average(g3['MURES'],weights=1/g3['MUERR']**2))
                data_hr_errs_hi.append(g3['MURES'].std()/np.sqrt(len(g3['MURES'])))
                data_c_mids_hi.append(n.mid)


        axsSFR.errorbar(data_c_mids_lo,data_hr_mids_lo,yerr=data_hr_errs_lo,marker='D',color=oldest,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR $\log(\mathrm{sSFR}/\mathrm{yr}^{-1})<11$')
        axsSFR.errorbar(data_c_mids_mid,data_hr_mids_mid,yerr=data_hr_errs_mid,marker='D',color=oldish,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR $-11\leq\log(\mathrm{sSFR}/\mathrm{yr}^{-1})<9.5$')

        axsSFR.errorbar(data_c_mids_hi,data_hr_mids_hi,yerr=data_hr_errs_hi,marker='D',color=youngest,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR $\log(\mathrm{sSFR}/\mathrm{yr}^{-1})>-9.5$')


    splits = {'lo':{'data_c_mids':data_c_mids_lo,'data_hr_mids':data_hr_mids_lo,'data_hr_errs':data_hr_errs_lo,'model_c_mids':model_c_mids_lo,'model_hr_mids':model_hr_mids_lo},
           'mid':{'data_c_mids':data_c_mids_mid,'data_hr_mids':data_hr_mids_mid,'data_hr_errs':data_hr_errs_mid,'model_c_mids':model_c_mids_mid,'model_hr_mids':model_hr_mids_mid},
           'hi':{'data_c_mids':data_c_mids_hi,'data_hr_mids':data_hr_mids_hi,'data_hr_errs':data_hr_errs_hi,'model_c_mids':model_c_mids_hi,'model_hr_mids':model_hr_mids_hi}
              }

    chisq = get_red_chisq_interp_split_multi(splits)
    axsSFR.text(-0.1,-0.25,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
    ''' axsSFR.errorbar(mvdf['avg_colour_log(sSFR)<-10.6,x_1<-0.3'],mvdf['avg_mures_log(sSFR)<-10.6,x_1<-0.3'],
        yerr=mvdf['stdm_mures_log(sSFR)<-10.6,x_1<-0.3'],marker='D',
        color=oldest,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data low sSFR; low $x_1$')
    axsSFR.errorbar(mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1<-0.3'],mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'],
        yerr=mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'],marker='o',
        color=oldish,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data mid sSFR; low $x_1$')

    axsSFR.errorbar(mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1>-0.3'],mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'],
        yerr=mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'],marker='s',
        color=youngish,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data mid sSFR; high $x_1$')

    axsSFR.errorbar(mvdf['avg_colour_log(sSFR)<-9.5,x_1>-0.3'],mvdf['avg_mures_log(sSFR)<-9.5,x_1>-0.3'],
        yerr=mvdf['stdm_mures_log(sSFR)<-9.5,x_1>-0.3'],marker='^',
        color=youngest,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data high sSFR; high $x_1$')
    obs = {'lo_lo':{'c':mvdf['avg_colour_log(sSFR)<-10.6,x_1<-0.3'].values,'hr':mvdf['avg_mures_log(sSFR)<-10.6,x_1<-0.3'].values,'hr_err':mvdf['stdm_mures_log(sSFR)<-10.6,x_1<-0.3'].values},
           'mid_lo':{'c':mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1<-0.3'].values,'hr':mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'].values,'hr_err':mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'].values},
           'mid_hi':{'c':mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1>-0.3'].values,'hr':mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'].values,'hr_err':mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'].values},
           'hi_hi':{'c':mvdf['avg_colour_log(sSFR)<-9.5,x_1>-0.3'].values,'hr':mvdf['avg_mures_log(sSFR)<-9.5,x_1>-0.3'].values,'hr_err':mvdf['stdm_mures_log(sSFR)<-9.5,x_1>-0.3'].values}
              }
    mod = {'lo_lo':{'c':model_c_mids_lo_lo ,'hr_mids':model_hr_mids_lo_lo},
          'mid_lo':{'c':model_c_mids_mid_lo ,'hr_mids':model_hr_mids_mid_lo} ,
          'mid_hi':{'c':model_c_mids_mid_hi ,'hr_mids':model_hr_mids_mid_hi} ,
          'hi_hi':{'c':model_c_mids_hi_hi ,'hr_mids':model_hr_mids_hi_hi}
              }
    chisq = get_red_chisq_interp_splitx1(obs,mod)
    axsSFR.text(-0.1,-0.25,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)'''
    axsSFR.set_xlabel('$c$',size=24)

    axsSFR.legend(fontsize=10,ncol=2,loc='upper center')
    #axMASS.set_title(sim.save_string + '_paper',size=20)
    axsSFR.set_ylim(-0.3,0.3)
    axsSFR.set_xlim(-0.19,0.3)
    axsSFR.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axsSFR.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    axsSFR.tick_params(which='both',right=True,top=True,labelsize=16)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_%s'%(sim.save_string + '_paper')+label_ext)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_%s'%(sim.save_string + '_paper')+label_ext+'.pdf')
    return chis
def get_red_chisq(obs,mod,err):

    chisq = np.nansum((obs - mod)**2/err**2)
    return chisq/len(err[~np.isnan(err)])
