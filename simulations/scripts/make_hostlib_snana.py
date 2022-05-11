import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
import seaborn as sns
import os
sns.set_color_codes(palette='colorblind')
import itertools
from des_sn_hosts.simulations import aura
sim = aura.Sim('/home/wiseman/code/des_sn_hosts/simulations/config/for_hostlib_new.yaml')
n_samples=100000
hostlib_df = pd.DataFrame()
palette = itertools.cycle(sns.color_palette('viridis',n_colors=len(sim.multi_df.z.unique())))
from tqdm import tqdm
idx = pd.IndexSlice
def interpolate_zdf(zdf,marr):
    '''Function to iterpolate SFH data frame onto a log-linear mass grid '''

    gb =zdf.groupby(pd.cut(zdf['mass'],bins=marr)).agg(np.mean)
    gb.dropna(subset=['mass'],inplace=True)
    gb.reset_index(drop=True,inplace=True)
    return gb
age_grid = np.arange(0,13.7,0.0005)
age_grid_index = ['%.4f'%a for a in age_grid]
#f,(axes)=plt.subplots(len(sim.multi_df.z.unique()),figsize=(16,25))
#axes = itertools.cycle(axes)
for z in tqdm(sim.multi_df.z.unique()):
    print(z)


    z_df = sim.multi_df.loc['%.3f' % z].copy()

    z_df['N_total'].replace(0., np.NaN, inplace=True)
    z_df.dropna(subset=['N_total'],inplace=True)
    #print('#########')
    #print(z)
    #print('N total min',z_df['N_total'].min())
    #print('Rate min',z_df['pred_rate_total'].min())
    z_df['N_SN_float'] = z_df['N_total'] / z_df['N_total'].min()  # Normalise the number of SNe so that the most improbable galaxy gets 1

    z_df['N_SN_int'] = z_df.loc[:,'N_SN_float'].astype(int)

    # Now we set up some index arrays so that we can sample masses properly
    #mav0_inds = z_df.loc[idx[:, '0.00000', :]].index
    resampled_df = pd.DataFrame()
    marr= np.logspace(6,11.6,100)
    for av in z_df.Av.unique():
        av_df =z_df.loc[idx[:, '%.5f'%av, :]]
        #print(av_df)
        av_df = interpolate_zdf(av_df,marr)
        resampled_df = resampled_df.append(av_df)
    #print(resampled_df.columns)
    Av_str = resampled_df['Av'].apply(lambda x: '%.5f'%x)
    mass_str = resampled_df['mass'].apply(lambda x: '%.2f'%x)
    new_zdf = resampled_df.set_index([mass_str,Av_str])

    m_inds = ['%.2f' % m for m in new_zdf['mass'].unique()]

    m_rates = []
    m_rates_float = []
    for m in m_inds:
        m_df = new_zdf.loc[m]
        mav_inds = (m, '%.5f' % (m_df.Av.unique()[0]))
        #print(new_zdf.loc[mav_inds,'N_SN_int'])
        m_rates.append(new_zdf.loc[mav_inds,'N_SN_int'])
        m_rates_float.append(new_zdf.loc[m,'N_SN_float'])
    c=next(palette)

    m_samples = np.random.choice(m_inds, p=m_rates / np.sum(m_rates), size=int(n_samples))
    # Now we have our masses, but each one needs some reddening. For now, we just select Av at random from the possible Avs in each galaxy
    # The stellar population arrays are identical no matter what the Av is.
    m_av0_samples = [(m, '%.5f' % (np.random.choice(new_zdf.loc[m].Av.values))) for m in m_samples]
    new_zdf['SN_ages'] = [age_grid for i in range(len(new_zdf))]
    new_zdf['SN_age_dist'] = [np.zeros(len(age_grid)) for i in range(len(new_zdf))]

    age_dists = []
    for n,g in z_df.groupby(pd.cut(z_df['mass'],bins=marr)):
        age_df = pd.DataFrame(index=age_grid_index)
        if len(g)>0:
            min_av = g.Av.astype(float).min()
            g_Av_0 =  g.loc[idx[:, '%.5f'%min_av, :]]
            for k in g_Av_0.index.unique():
                sub_gb = g_Av_0.loc[k]
                if type(sub_gb)==pd.DataFrame:
                    tf = sub_gb['t_f'].iloc[0]


                    j =np.random.randint(0,len(sub_gb))
                    split_z = os.path.split(self.config['hostlib_fn'])[1].split('z')
                    split_rv = os.path.split(self.config['hostlib_fn'])[1].split('rv')
                    ext = split_z[0]+'z_'+'%.2f_'%z+'rv'+split_rv[1][:-12]+'_%.1f'%tf+'_combined.dat'
                    new_fn = os.path.join(os.path.split(self.config['hostlib_fn'])[0],'SN_ages',ext)
                    sub_gb = pd.read_csv(new_fn,sep=' ',names=['SN_ages','SN_age_dist'])
                else:
                    tf = sub_gb['t_f']
                    split_z = os.path.split(self.config['hostlib_fn'])[1].split('z')
                    split_rv = os.path.split(self.config['hostlib_fn'])[1].split('rv')
                    ext = split_z[0]+'z_'+'%.2f_'%z+'rv'+split_rv[1][:-12]+'_%.1f'%tf+'_combined.dat'
                    new_fn = os.path.join(os.path.split(self.config['hostlib_fn'])[0],'SN_ages',ext)
                    sub_gb = pd.read_csv(new_fn,sep=' ',names=['SN_ages','SN_age_dist'])
                age_inds = ['%.4f'%a for a in sub_gb['SN_ages']]
                age_df.loc[age_inds,'%.2f'%(float(k))] = sub_gb['SN_age_dist'].values/np.nansum( sub_gb['SN_age_dist'].values)
            age_df.fillna(0,inplace=True)
            for av in g.Av.unique():
                age_dists.append(np.nanmean(age_df,axis=1))

        else:

            pass

    new_zdf.index.names = ['mass_index','Av_index']
    new_zdf.sort_values(by='mass',inplace=True)
    new_zdf['SN_age_dist']=age_dists
    # Now we sample from our galaxy mass distribution, given the expected rate of SNe at each galaxy mass

    gals_df = new_zdf.loc[m_av0_samples,['z','mass','ssfr','m_g','m_r','m_i','m_z','U', 'B', 'V', 'R', 'I','U_R','mean_age','Av','pred_rate_total']]

    sn_ages = [np.random.choice(new_zdf.loc[i,'SN_ages'],p=new_zdf.loc[i,'SN_age_dist']) for i in m_av0_samples] #/new_zdf.loc[i,'SN_age_dist'].sum()
    gals_df['SN_age'] = np.array(sn_ages)
    hostlib_df=hostlib_df.append(gals_df)
hostlib_df['a0_Sersic'] = ((np.array([np.max([0.185,np.random.normal(-0.18*m+5,0.3)]) for m in hostlib_df['m_r']]))*(hostlib_df['m_r']>17.5) )+ ((np.array([np.max([1,np.random.normal(3.5,2)]) for i in hostlib_df['m_r']])*(hostlib_df['m_r']<=17.5)))

hostlib_df['b0_Sersic'] = ((np.array([np.max([0.12,np.random.normal(-0.14*m+3.8,0.15)]) for m in hostlib_df['m_r']]))*(hostlib_df['m_r']>17.5) )+ ((np.array([np.max([1,np.random.normal(2.8,1)]) for i in hostlib_df['m_r']])*(hostlib_df['m_r']<=17.5)))
hostlib_df['n0_Sersic'] = 0.5
hostlib_df['obs_gr'] = hostlib_df['m_g'] - hostlib_df['m_r']
hostlib_df['LOGMASS_ERR'] = 0
hostlib_df['LOG_sSFR_ERR'] = 0
hostlib_df['LOG_SFR'] = hostlib_df['mass']*hostlib_df['ssfr']
hostlib_df['LOG_SFR_ERR'] = 0
hostlib_df['a_rot'] = 0
hostlib_df['VARNAMES:'] = 'GAL'


hostlib_df.reset_index(inplace=True)
hostlib_df.reset_index(inplace=True)

hostlib_df.rename(columns={'index:':'GALID','index':'GALID','m_g':'g_obs','m_r':'r_obs','m_i':'i_obs','m_z':'z_obs'},inplace=True)


field_centres = np.array([[54.2743, -27.1116],
                [54.2743, -29.0844],
                [52.6484, -28.1000],
                 [7.8744, -43.0096],
                 [9.5000, -43.9980],
                 [42.82,  0.0000],
                 [41.1944, -0.9884],
                 [34.4757, -4.9295],
                 [35.6645, -6.4121],
                [36.4500, -4.6000]])
fields = np.random.choice(range(10),size=len(hostlib_df))


radecs = np.random.uniform(field_centres[fields]-1.5,field_centres[fields]+1.5)

f,ax=plt.subplots()
ax.scatter(radecs[:,0],radecs[:,1])

hostlib_df['RA'] = radecs[:,0]
hostlib_df['DEC'] = radecs[:,1]

hostlib_df.rename(columns={'z':'ZTRUE','mass':'LOGMASS','ssfr':'LOG_sSFR'},inplace=True)
hostlib_df['ZERR'] = 0
hostlib_df=hostlib_df[['VARNAMES:','GALID', 'RA','DEC','ZTRUE', 'g_obs', 'r_obs', 'i_obs', 'z_obs',
            'n0_Sersic','a0_Sersic', 'b0_Sersic', 'a_rot',
            'LOGMASS','LOGMASS_ERR','LOG_SFR', 'LOG_SFR_ERR', 'LOG_sSFR','obs_gr',
       'U', 'B', 'V', 'R', 'I', 'U_R', 'mean_age', 'Av', 'pred_rate_total',
       'SN_age',     ]]
hostlib_df.to_hdf(os.path.join(sim.root_dir,'sims/hostlibs/Phil_Hostlib.h5'),key='main',index=False)
hostlib_df.to_csv(os.path.join(sim.root_dir,'sims/hostlibs/Phil_Hostlib.csv'),)
