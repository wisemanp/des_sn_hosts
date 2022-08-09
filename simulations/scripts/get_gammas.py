import numpy as np
import pandas as pd
import sys
from des_sn_hosts.simulations import aura
import hubblefit
from yaml import safe_load as yload


def stderr(x):
    if len(x)>0:
        print(np.std(x),np.sqrt(len(x)))
        return np.std(x)#/np.sqrt(len(x))
    else:
        return np.NaN

def my_agg(x):
    if len(x)>0:
        names = {'MU': (np.average(x['MU'],weights=x['MUERR'])),'MUERR':(np.std(x['MU'])/np.sqrt(len(x))),'zHD':(x['zHD'].mean())}
        #names = {'MU': (np.average(x['MU'],weights=x['MUERR'])),'MUERR':0.05,'zHD':(x['zHD'].mean())}
    else:
        names = {'MU': np.NaN,'MUERR':np.NaN,'zHD':np.NaN}
    return pd.Series(names, index=['zHD','MU','MUERR'])

    def measure_steps_nobbc(data,tracer_dict):
        steps,errs = [],[]
        steps_nogamma,errs_nogamma = [],[]
        for counter,(tracer,this_tracer_dict) in enumerate(tracer_dict.items()):
            #print(tracer)
            tracer_name = this_tracer_dict['tracer_name']
            ttype = this_tracer_dict['ttype']
            split = this_tracer_dict['split']
            ttag = this_tracer_dict['ttag']
            print(tracer,this_tracer_dict)
            pop_a = data[data[tracer]<split]
            pop_b = data[data[tracer]>=split]

            # Mickael Fit, fit gamma
            data_m = data.rename(columns={'CID':'name','mB':'mb','mB_err':'mb.err','x1ERR':'x1.err','cERR':'c.err','HOST_LOGMASS_ERR':'HOST_LOGMASS.err','zHD':'zcmb','zHDERR':'zcmb.err'})
            #data_m['HOST_LOGMASS.err']= np.ones_like(data_m['HOST_LOGMASS'])*0.1
            #data_m['mean_ages.err'] = np.ones_like(data_m['mean_ages'])*50
            data_m['P_tracer'] = data_m[tracer].apply(lambda x: 1 if x>split else 0)
            cols = ['name','zcmb','zcmb.err','mb','mb.err','x1','x1.err','c','c.err','P_tracer']

            hfit = hubblefit.get_hubblefit(data_m[cols],corr=['x1','c','P_tracer'],use_minuit=False,build=True,verbose=False,cosmo=FlatLambdaCDM(70,0.3))
            hfit.use_minuit=False
            hfit.model.M0_guess=-19
            hfit.model.sigmaint_guess=0.1
            hfit.model.a1_guess=-0.15
            hfit.model.a2_guess = 2.5
            hfit.model.a3_guess = -0.1
            hfit.fit()
            gamma_mickael = -1*hfit.model.standardization_coef['a3']
            steps.append(gamma_mickael )
            errs.append(0.01)
            data_m['mu_res'],data_m['mu_res_err'] = hfit.get_hubbleres()
            data_m['MU'] = data_m['mu_res']+cosmo.distmod(data_m['zcmb']).value
            tracer_dict[tracer]['HD'] = data_m[['zcmb','MU','mu_res_err']]


            # Mickael Fit, gamma after
            hfit = hubblefit.get_hubblefit(data_m[cols],corr=['x1','c'],use_minuit=False,build=True,verbose=False)
            hfit.use_minuit=False
            hfit.model.M0_guess=-19
            hfit.model.sigmaint_guess=0.1
            hfit.model.a1_guess=-0.15
            hfit.model.a2_guess = 2.5
            hfit.fit()
            data_m['mu_res'],data_m['mu_res_err'] = hfit.get_hubbleres()
            pop_a = data_m[data_m[tracer]<split]
            pop_b = data_m[data_m[tracer]>=split]

            step = np.average(pop_a['mu_res'],weights=pop_a['mu_res_err']) -  np.average(pop_b['mu_res'],weights=pop_b['mu_res_err'])
            error = np.sqrt((np.sqrt(np.cov(pop_a['mu_res'],aweights=pop_a['mu_res_err']))/np.sqrt(len(pop_a['mu_res'])))**2 + (np.sqrt(np.cov(pop_b['mu_res'],aweights=pop_b['mu_res_err']))/np.sqrt(len(pop_b['mu_res'])))**2)
            steps_nogamma.append(step)
            errs_nogamma.append(error)
        return steps,errs,steps_nogamma,errs_nogamma, tracer_dict





conf_path = sys.argv[1]
with open(conf_path,'r') as f:
    config =  yload(f)
sim = aura.Sim(config['sim_config'])

sim.config['mB_model']['params']['age_step']['mag']=config['age_step']
sim.config['mB_model']['params']['mass_step']['mag']=config['mass_step']

if config['rv_tracer']=='mass':
    hi='low'
    lo='high'
elif config['rv_tracer']=='age':
    hi='young'
    lo='old'
sim.config['SN_rv_model']['params'] = {'rv_'%hi: config['rv_hi'],
  'rv_%s'%lo: config['rv_lo'],
  'rv_sig_%s'%hi: 1.0,
  'rv_sig_%s'%lo: 1.0,
  '%s_split'config['rv_tracer']: config['rv_step'],
  'rv_min': config['rv_min']}
n_samples=config['n_samples']

## Do the simulations

zs = np.linspace(0.0,1.2,1000)
zs_cubed = zs**3.
numbers = np.random.choice(zs,p=zs_cubed/np.sum(zs_cubed),size=n_samples)
zarr = np.arange(0.14,1.2,0.1)
n_samples_arr = sim._get_z_dist(numbers,n=n_samples,frac_low_z=0,zbins=zarr+0.02)
highz_fn = '/media/data3/wiseman/des/AURA/sims/SNe/Briday/DES_BS20_age_Rv_step_3Gyr_age_x1_beta_1.14_quenched_bursty_highz_%i_SN_sim.h5'%config['simno']
sim.sample_SNe(zarr,n_samples_arr,savepath=highz_fn)
zarr = np.arange(0.0105,0.14,0.01)
n_samples_arr = sim._get_z_dist(numbers,n=n_samples,frac_low_z=0,zbins=zarr+0.02)
lowz_fn = highz_fn.replace('high','low')
sim.sample_SNe(zarr,n_samples_arr,savepath='/media/data3/wiseman/des/AURA/sims/SNe/Briday/DES_BS20_age_Rv_step_3Gyr_age_x1_beta_1.14_quenched_bursty_lowz_%i_SN_sim.h5'%config['simno'])


sim.load_sim(lowz_fn)
sim.sim_df =sim.sim_df#.sample(500)
data_midz = pd.read_hdf(highz_fn)
data_midz = data_midz#.sample(1500)
sim.sim_df = sim.sim_df.append(data_midz)
sim.sim_df.reset_index(inplace=True)

sim.sim_df.rename(columns={'mBERR':'mB_err','z':'zHD','x1_err':'x1ERR','c_err':'cERR','index':'CID'},inplace=True)
#sim.sim_df =sim.sim_df[sim.sim_df['eff_mask']==1]
sim.sim_df=sim.sim_df[(sim.sim_df['x1']>-3)&(sim.sim_df['x1']<3)&(sim.sim_df['c']<0.3)&(sim.sim_df['c']>-0.3)]
sim.sim_df['HOST_LOGMASS'] =np.log10(sim.sim_df['mass'])
sim.sim_df = sim.sim_df[sim.sim_df['HOST_LOGMASS']>8]
sim.sim_df['HOST_LOG_sSFR'] =np.log10(sim.sim_df['ssfr'])
sim.sim_df['zHDERR'] =0.0001
data = sim.sim_df#.sample(1500)
data['HOST_LOG_sSFR'] = -1*data['HOST_LOG_sSFR']

steps,errs,steps_nogamma,errs_nogamma,tracer_dict = measure_steps_nobbc(data,tracer_dict = {
    'HOST_LOGMASS':{'tracer_name':'Mass','ttype':'Measured' , 'split':10 ,'ttag':'hostmass' },
    'HOST_LOG_sSFR':{'tracer_name':'sSFR (Phot)','ttype':'Measured' , 'split':10.3 ,'ttag':'hostssfr' },
    'U-R':{'tracer_name':'U-R','ttype': 'Measured', 'split':1 ,'ttag': 'hostUR'},
    'mean_ages':{'tracer_name':'Galaxy Age','ttype':'Model' , 'split':3000 ,'ttag':'hostage' },
    'SN_age':{'tracer_name':'SN Age','ttype':'Model' , 'split':0.75 ,'ttag':'snage' }
})

savearr = np.array([steps,errs,steps_nogamma,errs_nogamma])
np.savetxt('/media/data3/wiseman/des/AURA/sims/SNe/Briday_Steps_%s_lowz.dat'%config['simno'],savearr)

simno=122
ztype='all'
for k,v in tracer_dict.items():
    v['HD'].to_hdf('/media/data3/wiseman/des/AURA/sims/SNe/wfit/HD_%i_%sz.h5'%(simno,ztype),key=k)
print('Saved to /media/data3/wiseman/des/AURA/sims/SNe/wfit/HD_%i_%sz.h5'%(simno,ztype))

for tracer in tracer_dict.keys():

  for_wfit = tracer_dict[tracer]['HD']
  for_wfit.rename(columns={'mu_res_err':'MUERR','zcmb':'zHD'},inplace=True)
  for_wfit = for_wfit[['zHD','MU','MUERR',]]
  for_wfit['VARNAMES:']='SN:'
  gbmeasureage =for_wfit.groupby(pd.cut(for_wfit['zHD'],bins=zarr+0.02)).apply(my_agg)
  gbmeasureage['VARNAMES:'] = 'SN:'
  gbmeasureage.dropna(axis=0)[['VARNAMES:','zHD','MU','MUERR',]].to_csv('/media/data3/wiseman/des/AURA/sims/SNe/wfit/Briday_Measure%s_%i_lowz.FITRES'%(tracer,config['simno']),index=False, sep=' ', quoting=csv.QUOTE_NONE, quotechar="",escapechar=" ")
