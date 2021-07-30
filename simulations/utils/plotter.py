'''A set of routines for plotting AURA simulations'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_color_codes(palette='colorblind')
import itertools
import os
import pandas as pd
import pickle
from .HR_functions import calculate_step, get_red_chisq_interp


aura_dir = os.environ['AURA_DIR']
des5yr = pd.read_csv(os.path.join(aura_dir,'data','df_after_cuts_z0.6_UR1.csv'))
lisa_data = pickle.load(open(os.path.join(aura_dir,'data','des5yr_hosts.pkl'),'rb'))
plt.style.use('default')
sns.set_context('paper')
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.figsize': [16,9]})
plt.rcParams.update({'xtick.direction':'in'})
plt.rcParams.update({'ytick.direction':'in'})
split_colour_1 = '#f28500'
split_colour_2 = '#8500f2'



def plot_galaxy_properties(sim):

    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,4),sharey=True)
    cm =ax1.scatter(sim.flux_df['mean_age']/1000,sim.flux_df['U_R'],c=sim.flux_df['Av'],alpha=0.3)
    ax1.set_xscale('log')
    #ax1.set_xlabel('$\log (M_*/M_{\odot})$',size=20)
    ax1.set_xlabel('Mean Stellar Age (Gyr)',size=20)
    ax1.set_ylabel('U-R',size=20)
    ax1.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])


    cm =ax2.scatter(sim.flux_df['ssfr'],sim.flux_df['U_R'],c=sim.flux_df['Av'],alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xlabel('$\log (sSFR)$ yr$^{-1}$',size=20)
    #ax2.set_ylabel('U-R',size=20)
    ax2.tick_params(which='both',labelsize=14,right=True,top=True)

    #ax3.set_xscale('log')
    ax3.set_xlabel('$\log (M_*/M_{\odot})$',size=20)

    #ax3.set_ylabel('U-R',size=20)
    ax3.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
    cm =ax3.scatter(np.log10(sim.flux_df['mass']),sim.flux_df['U_R'],c=sim.flux_df['Av'],alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='vertical',ax=ax3,#shrink=0.7
                   )
    cb.set_label('$A_V$',size=20,
                )

    plt.savefig(sim.fig_dir +'color_vs_host_params')
    # plot U-R v mass only
    f,ax=plt.subplots(figsize=(8,6.5))
    #ax3.set_xscale('log')
    ax.set_xlabel('$\log (M_*/M_{\odot})$',size=20)

    #ax3.set_ylabel('U-R',size=20)
    ax.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
    from matplotlib.colors import ListedColormap
    cm =ax.scatter(np.log10(sim.flux_df['mass']),sim.flux_df['U_R'],c=sim.flux_df['Av'],alpha=0.3,
                   cmap=ListedColormap(sns.color_palette('viridis',n_colors=20).as_hex()))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='horizontal',)#shrink=0.7)
    ax.set_ylabel('U-R',size=20)
    cb.set_label('$A_V$',size=20,
                )

    plt.tight_layout()
    lisa_colours = pd.read_csv(os.path.join(aura_dir,'data','5yr-massUR.csv'),index_col=0)

    ax.errorbar(lisa_colours['Host Mass'],lisa_colours['Host U-R'],
                xerr=lisa_colours['Host Mass error'],yerr=lisa_colours['Host U-R error'],
                linestyle='none',marker='+',label='DES U-R global')

    #ax.scatter(lisa_colours['Host Mass'],lisa_colours['Host U-R']+0.58,
    #
    #           marker='+',label='DES U-R global',c=lisa_colours['Host Mass'],cmap='gist_rainbow')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim(-0.5,3)
    plt.savefig(sim.fig_dir +'U-R_vs_data')


def plot_cs(sim,df):
    f,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
    df['logmass'] = np.log10(df['mass'])
    ax1.scatter(df['logmass'],df['c'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis',label='Sim')

    for counter, (n,g) in enumerate(df.groupby(pd.cut(df['logmass'],bins=np.linspace(8,12,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'Sim Mean'
            ax1.scatter(n.mid,g['c'].mean(),color='c',edgecolor='w',linewidth=1,marker='D',s=100,label=label)
            ax1.errorbar(n.mid,g['c'].mean(),yerr=np.sqrt(np.mean(g['c']**2)),c='c',marker=None,ls='none')
    for counter, (n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['Host Mass'],bins=np.linspace(8,12,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'DES5YR Mean'
            ax1.scatter(n.mid,g['c'].mean(),color='m',edgecolor='w',linewidth=1,marker='s',s=100,label=label)
            ax1.errorbar(n.mid,g['c'].mean(),yerr=np.sqrt(np.mean(g['c']**2)),c='m',marker=None,ls='none')

    ax1.legend()
    cm=ax2.scatter(df['U-R'],df['c'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis',label='Sim')

    for counter, (n,g) in enumerate(df.groupby(pd.cut(df['U-R'],bins=np.linspace(-0.5,2.5,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'Sim Mean'
            ax2.scatter(n.mid,g['c'].mean(),color='c',edgecolor='w',linewidth=1,marker='D',s=100,label=label)
            ax2.errorbar(n.mid,g['c'].mean(),yerr=np.sqrt(np.mean(g['c']**2)),c='c',marker=None,ls='none')
    for counter, (n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['Host U-R'],bins=np.linspace(-0.5,2.5,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'DES5YR Mean'
            ax2.scatter(n.mid,g['c'].mean(),color='m',edgecolor='w',linewidth=1,marker='s',s=100,label=label)
            ax2.errorbar(n.mid,g['c'].mean(),yerr=np.sqrt(np.mean(g['c']**2)),c='m',marker=None,ls='none')
    ax2.legend()


    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='vertical',ax=ax2)#shrink=0.7)
    ax1.set_ylabel('$c$',size=20)
    cb.set_label('$A_V$',size=20)
    for ax in [ax1,ax2]:
        ax.tick_params(which='both',direction='in',top=True,right=True)
    ax1.set_xlabel('Stellar Mass',size=20)
    ax2.set_xlabel('$U-R$',size=20)
    ax2.set_xlim(0,2)
    ax1.set_xlim(7.8,11.8)
    ax1.set_ylim(-0.3,0.3)
    plt.savefig(sim.fig_dir +'SN_c_hosts_%s'%sim.save_string)
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.hist(df['c'],bins=np.linspace(-0.3,0.3,100),histtype='step',density=True,label='Sim',lw=3,color='c')
    ax.hist(des5yr['c'],density=True,bins=20,histtype='step',label='DES5YR',lw=3,color='m')
    #ax.hist(pantheon['c'],density=True,bins=25,histtype='step',color='y',label='Obs Pantheon',lw=3)
    ax.legend()
    ax.set_xlabel('c',size=20)
    plt.savefig(sim.fig_dir +'SN_c_hist_%s'%sim.save_string)
    # get chi2
    counts,bin_edges =np.histogram(des5yr['c'],bins=np.linspace(-3,3,20),density=False)
    simcounts,simbins = np.histogram(sim.sim_df['c'],bins=np.linspace(-3,3,20),density=False)
    simcounts = simcounts/(len(sim.sim_df)/len(des5yr))
    intervals = poisson_conf_interval(counts,interval='root-n-0').T
    yplus= intervals[:,1] -counts
    chi2 = get_red_chisq(counts,simcounts,yplus)
    return chi2
def plot_x1s(sim,df,return_chi=True):
    f,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
    df['logmass'] = np.log10(df['mass'])
    ax1.scatter(df['logmass'],df['x1'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')

    for counter, (n,g) in enumerate(df.groupby(pd.cut(df['logmass'],bins=np.linspace(8,12,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'Sim Mean'
            ax1.scatter(n.mid,g['x1'].mean(),color='c',edgecolor='w',linewidth=1,marker='D',s=100,label=label)
            ax1.errorbar(n.mid,g['x1'].mean(),yerr=np.sqrt(np.mean(g['x1']**2)),c='c',marker=None,ls='none')
    for counter, (n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['Host Mass'],bins=np.linspace(8,12,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'DES5YR Mean'
            ax1.scatter(n.mid,g['x1'].mean(),color='m',edgecolor='w',linewidth=1,marker='s',s=100,label=label)
            ax1.errorbar(n.mid,g['x1'].mean(),yerr=np.sqrt(np.mean(g['x1']**2)),c='m',marker=None,ls='none')

    cm=ax2.scatter(df['U-R'],df['x1'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis',label='Sim')

    for counter, (n,g) in enumerate(df.groupby(pd.cut(df['U-R'],bins=np.linspace(-0.5,2.5,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'Sim Mean'
            ax2.scatter(n.mid,g['x1'].mean(),color='c',edgecolor='w',linewidth=1,marker='D',s=100,label=label)
            ax2.errorbar(n.mid,g['x1'].mean(),yerr=np.sqrt(np.mean(g['x1']**2)),c='c',marker=None,ls='none')
    for counter, (n,g) in enumerate(des5yr.groupby(pd.cut(des5yr['Host U-R'],bins=np.linspace(-0.5,2.5,30)))):
        if len(g)>0:
            label=None
            if counter==0:
                label = 'DES5YR Mean'
            ax2.scatter(n.mid,g['x1'].mean(),color='m',edgecolor='w',linewidth=1,marker='s',s=100,label=label)
            ax2.errorbar(n.mid,g['x1'].mean(),yerr=np.sqrt(np.mean(g['x1']**2)),c='m',marker=None,ls='none')
    ax1.legend()

    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='vertical',ax=ax2)#shrink=0.7)
    ax1.set_ylabel('$x_1$',size=20)
    cb.set_label('$A_V$',size=20,
                )
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    for ax in [ax1,ax2]:
        ax.tick_params(which='both',direction='in',top=True,right=True)
    ax1.set_xlabel('Stellar Mass',size=20)
    ax1.set_xlim(7.8,11.8)
    ax1.set_ylim(-3,3)
    ax2.set_xlabel('$U-R$',size=20)
    ax2.set_xlim(0,2.5)
    plt.savefig(sim.fig_dir +'SN_x1_hosts_%s'%sim.save_string)
    f,ax=plt.subplots(figsize=(8,6.5))
    hist=ax.hist(df['x1'],bins=np.linspace(-3,3,100),density=True,label='Simulation',histtype='step',lw=3)
    ax.set_xlabel('$x_1$',size=20)
    ax.hist(des5yr['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='DES5YR')
    #ax.hist(pantheon['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='Pantheon')
    ax.legend()
    plt.savefig(sim.fig_dir +'SN_x1_hist_%s'%sim.save_string)
    #calculate the reduced chi-squared
    counts,bin_edges =np.histogram(des5yr['x1'],bins=np.linspace(-3,3,20),density=False)
    simcounts,simbins = np.histogram(sim.sim_df['x1'],bins=np.linspace(-3,3,20),density=False)
    simcounts = simcounts/(len(sim.sim_df)/len(des5yr))
    intervals = poisson_conf_interval(counts,interval='root-n-0').T
    yplus= intervals[:,1] -counts
    chi2 = get_red_chisq(counts,simcounts,yplus)
    return chi2
def plot_samples(sim,zmin=0,zmax=1.2,x1=True,c=True,hosts=True):
    plot_df=sim.sim_df[(sim.sim_df['z']>zmin)&(sim.sim_df['z']<zmax)]
    if x1:
        plot_x1s(sim,plot_df)
    if c:
        plot_cs(sim,plot_df)

def plot_mu_res(sim,obs=True,label_ext='',colour_split=1,mass_split=1E+10,return_chi=True):
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.set_title(sim.save_string,size=20)
    g1 = sim.sim_df[sim.sim_df['mass']>1E+10]
    g2 = sim.sim_df[sim.sim_df['mass']<=1E+10]

    labelhi='High mass Host'
    labello='Low mass Host'
    ax.scatter(g1['c'],g1['mu_res'],c=split_colour_2,marker='o',s=50,alpha=0.3,label=labelhi,edgecolor='w',linewidth=0.5)
    ax.scatter(g2['c'],g2['mu_res'],c=split_colour_1,marker='o',s=50,alpha=0.3,edgecolor='w',linewidth=0.5,label=labello,)

    ax.set_xlabel('$c$',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.legend(fontsize=16)

    f,ax=plt.subplots(figsize=(8,6.5))
    ax.set_title(sim.save_string,size=20)
    ax.hist(sim.sim_df['mu_res'],bins=100)
    ax.set_xlabel('$\mu_{\mathrm{res}}$',size=20)

    plt.savefig(sim.fig_dir +'HR_hist_%s'%(sim.save_string)+label_ext)
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.set_title(sim.save_string,size=20)
    cb=ax.scatter(sim.sim_df['U-R'],sim.sim_df['mu_res'],alpha=0.3,c=sim.sim_df['c'],cmap='rainbow')
    plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,30))):
        if len(g)>0:
            ax.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='r',marker='D',s=100)
            ax.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,3))):
        print(n.mid)
        ax.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='c',marker='s',markersize=20,ls='none')

    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['U-R'],1)
    ax.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=ax.transAxes)
    ax.set_xlabel('$U-R$',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.set_title(sim.save_string,size=20)
    ax.set_ylim(-0.3,0.3)
    ax.set_xlim(-0.5,2.5)

    plt.savefig(sim.fig_dir +'HR_vs_UR_scatter_%s'%(sim.save_string)+label_ext)
    f,ax=plt.subplots(figsize=(8,6.5))
    sim.sim_df['logmass'] = np.log10(sim.sim_df['mass'])
    cb=ax.scatter(sim.sim_df['logmass'],sim.sim_df['mu_res'],alpha=0.3,c=sim.sim_df['c'],cmap='rainbow')
    plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,30))):
        if len(g)>0:
            ax.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='r',marker='D',s=100)
            ax.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,3))):
        print(n.mid)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='c',marker='s',markersize=20,)
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['logmass'],10)
    ax.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=ax.transAxes)
    ax.set_xlabel('$\log(M_*/M_{\odot})$',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.set_xlim(7.5,12)
    ax.set_ylim(-0.3,0.3)
    ax.set_title(sim.save_string,size=20)

    plt.savefig(sim.fig_dir +'HR_vs_mass_scatter_%s'%(sim.save_string)+label_ext)
    f,ax=plt.subplots(figsize=(8,6.5))
    sim.sim_df['logSN_age'] = np.log10(sim.sim_df['SN_age'])
    cb=ax.scatter(sim.sim_df['logSN_age'],sim.sim_df['mu_res'],alpha=0.3,c=sim.sim_df['c'],cmap='rainbow')
    plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logSN_age'],bins=np.linspace(-1.5,1,30))):
        if len(g)>0:
            ax.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='r',marker='D',s=100)
            ax.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logSN_age'],bins=np.linspace(-1.5,1,3))):
        print(n.mid)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='c',marker='s',markersize=20,ls='none')
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['logSN_age'],0.75)
    ax.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=ax.transAxes)
    ax.set_xlabel('$\log$ (SN Age (Gyr))',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.set_title(sim.save_string,size=20)
    ax.set_ylim(-0.3,0.3)

    plt.savefig(sim.fig_dir +'/HR_vs_age_scatter_%s'%(sim.save_string)+label_ext)
    chis = []
    fMASS,axMASS=plt.subplots(figsize=(8,6.5))
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
    if obs:
        low =lisa_data['global_mass']['low']
        high=lisa_data['global_mass']['high']
        axMASS.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr=low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})<10$')
        axMASS.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr=high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})>10$')
        chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        axMASS.text(-0.2,-0.05,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)
    axMASS.set_xlabel('$c$',size=20)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axMASS.legend(fontsize=13)
    axMASS.set_title(sim.save_string,size=20)
    axMASS.set_ylim(-0.2,0.2)

    plt.savefig(sim.fig_dir +'HR_vs_c_split_mass_%s'%(sim.save_string)+label_ext)
    fAGE,axAGE=plt.subplots(figsize=(8,6.5))
    axAGE.set_title(sim.save_string,size=20)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi =[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['SN_age']>0.8]
            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['SN_age']<=0.8]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass

    axAGE.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Young Progenitor')
    axAGE.plot(model_c_mids_lo ,np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),c=split_colour_1,lw=0.5,ls=':')
    axAGE.plot(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),c=split_colour_1,lw=0.5,ls=':')
    axAGE.plot(model_c_mids_hi ,model_hr_mids_hi,c=split_colour_2,lw=3,label='Model Old Progenitor',ls='--')
    axAGE.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)+np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    axAGE.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    axAGE.set_xlabel('$c$',size=20)
    axAGE.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axAGE.legend(fontsize=13)
    axAGE.set_ylim(-0.2,0.2)

    plt.savefig(sim.fig_dir +'HR_vs_c_split_SN_age_%s'%(sim.save_string)+label_ext)


    fmeanAGE,axmeanAGE=plt.subplots(figsize=(8,6.5))
    axmeanAGE.set_title(sim.save_string,size=20)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi =[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['mean_age']>3000]


            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['mean_age']<=3000]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass
    axmeanAGE.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Young Host')
    axmeanAGE.plot(model_c_mids_lo ,np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),c=split_colour_1,lw=0.5,ls=':')
    axmeanAGE.plot(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),c=split_colour_1,lw=0.5,ls=':')
    axmeanAGE.plot(model_c_mids_hi ,model_hr_mids_hi,c=split_colour_2,lw=3,label='Model Old Host',ls='--')
    axmeanAGE.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)+np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    axmeanAGE.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    axmeanAGE.set_xlabel('$c$',size=20)
    axmeanAGE.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axmeanAGE.legend(fontsize=13)
    axmeanAGE.set_ylim(-0.2,0.2)

    plt.savefig(sim.fig_dir +'HR_vs_c_split_age_%s'%(sim.save_string)+label_ext)

    fUR,axUR=plt.subplots(figsize=(8,6.5))
    axUR.set_title(sim.save_string,size=20)


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

    if obs:
        low =lisa_data['global_U-R']['low']
        high=lisa_data['global_U-R']['high']
        chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        axUR.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr =low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
        axUR.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr =high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
        axUR.text(-0.2,-0.05,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        chis.append(chisq)
    axUR.set_xlabel('$c$',size=20)
    axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axUR.legend(fontsize=13)
    axUR.set_ylim(-0.2,0.2)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_UR_%s'%(sim.save_string)+label_ext)
    return chis
def plot_rms(sim,label_ext='',colour_split=1,mass_split=1E+10):

    # Mass
    fMASSrms,(axMASSrms_all,axMASSrms)=plt.subplots(2,figsize=(8,6.5),sharex=True)
    model_c_mids_all , model_rms_mids_all , model_rms_errs_all =[],[],[]
    obs_c_mids_all , obs_rms_mids_all , obs_rms_errs_all =[],[],[]

    model_c_mids_lo , model_rms_mids_lo , model_rms_errs_lo , model_c_mids_hi , model_rms_mids_hi ,  model_rms_errs_hi =[],[],[],[],[],[]
    obs_c_mids_lo , obs_rms_mids_lo , obs_rms_errs_lo , obs_c_mids_hi , obs_rms_mids_hi ,  obs_rms_errs_hi =[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        #try:
            obs = des5yr[(des5yr['c']>n.left)&(des5yr['c']<n.right)]

            # All hosts
            obs_rms_mids_all.append(np.sqrt(np.mean(obs['cal residual']**2)))
            obs_rms_errs_all.append(obs['cal residual'].std()/np.sqrt(len(obs['cal residual'])))

            model_rms_mids_all.append(np.sqrt(np.mean(g['mu_res']**2)))
            model_rms_errs_all.append(g['mu_res'].std()/np.sqrt(len(g['mu_res'])))
            model_c_mids_all.append(n.mid)
            obs_c_mids_all.append(n.mid)

            # High mass
            obshi = obs[obs['Host Mass']>np.log10(mass_split)]
            g1 = g[g['mass']>mass_split]
            model_rms_mids_hi.append(np.sqrt(np.mean(g1['mu_res']**2)))
            model_rms_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))

            obs_rms_mids_hi.append(np.sqrt(np.mean(obshi['cal residual']**2)))
            obs_rms_errs_hi.append(obshi['cal residual'].std()/np.sqrt(len(obshi['cal residual'])))

            model_c_mids_hi.append(n.mid)
            obs_c_mids_hi.append(n.mid)


            # Low mass

            g2 = g[g['mass']<=mass_split]
            model_rms_mids_lo.append(np.sqrt(np.mean(g2['mu_res']**2)))
            model_rms_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))

            obslo = obs[obs['Host Mass']<np.log10(mass_split)]
            obs_rms_mids_lo.append(np.sqrt(np.mean(obslo['cal residual']**2)))
            obs_rms_errs_lo.append(obslo['cal residual'].std()/np.sqrt(len(obslo['cal residual'])))

            model_c_mids_lo.append(n.mid)
            obs_c_mids_lo.append(n.mid)
        #except:
           #pass
    axMASSrms_all.plot(model_c_mids_all ,model_rms_mids_all,c='g',lw=3,label='Model All')
    axMASSrms_all.fill_between(model_c_mids_all ,np.array(model_rms_mids_all)-np.array(model_rms_errs_all),np.array(model_rms_mids_all)+np.array(model_rms_errs_all),color='g',lw=0.5,ls=':',alpha=0.3)
    axMASSrms_all.errorbar(obs_c_mids_all ,obs_rms_mids_all,yerr=obs_rms_errs_all,marker='D',color='k',linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR')


    axMASSrms.plot(model_c_mids_lo ,model_rms_mids_lo,c=split_colour_1,lw=3,label='Model Low Mass')
    axMASSrms.fill_between(model_c_mids_lo ,np.array(model_rms_mids_lo)-np.array(model_rms_errs_lo),np.array(model_rms_mids_lo)+np.array(model_rms_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)

    axMASSrms.plot(model_c_mids_hi ,model_rms_mids_hi,c=split_colour_2,lw=3,label='Model High Mass',ls='--')
    axMASSrms.fill_between(model_c_mids_hi ,np.array(model_rms_mids_hi)-np.array(model_rms_errs_hi),np.array(model_rms_mids_hi)+np.array(model_rms_errs_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)
    #axMASS.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')

    axMASSrms.errorbar(obs_c_mids_lo ,obs_rms_mids_lo,yerr=obs_rms_errs_lo,marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})<10$')
    axMASSrms.errorbar(obs_c_mids_hi ,obs_rms_mids_hi,yerr=obs_rms_errs_hi,marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $\log(M_*/M_{\odot})>10$')
    #chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_rms_mids_lo,model_c_mids_hi,model_rms_mids_hi)
    #axMASSrms.text(-0.2,-0.05,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
    plt.subplots_adjust(hspace=0,wspace=0)
    axMASSrms_all.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axMASSrms_all.tick_params(right=True,top=True,direction='in',labelsize=14)
    axMASSrms_all.set_ylim(0,0.4)

    axMASSrms_all.set_ylabel('RMS ($\mu_{\mathrm{res}}$)',size=20,)
    axMASSrms_all.legend(fontsize=13)
    axMASSrms.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axMASSrms.tick_params(right=True,top=True,direction='in',labelsize=14)
    axMASSrms.set_xlabel('$c$',size=20)
    axMASSrms.set_ylabel('RMS ($\mu_{\mathrm{res}}$)',size=20,)
    axMASSrms.legend(fontsize=13)
    axMASSrms_all.set_title(sim.save_string,size=20)
    axMASSrms.set_ylim(0,0.4)
    plt.savefig(sim.fig_dir +'HR_rms_c_split_mass_%s'%(sim.save_string)+label_ext)

    # U-R
    fURrms,(axURrms_all,axURrms)=plt.subplots(2,figsize=(8,6.5),sharex=True)
    model_c_mids_all , model_rms_mids_all , model_rms_errs_all =[],[],[]
    obs_c_mids_all , obs_rms_mids_all , obs_rms_errs_all =[],[],[]

    model_c_mids_lo , model_rms_mids_lo , model_rms_errs_lo , model_c_mids_hi , model_rms_mids_hi ,  model_rms_errs_hi =[],[],[],[],[],[]
    obs_c_mids_lo , obs_rms_mids_lo , obs_rms_errs_lo , obs_c_mids_hi , obs_rms_mids_hi ,  obs_rms_errs_hi =[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        #try:
            obs = des5yr[(des5yr['c']>n.left)&(des5yr['c']<n.right)]

            # All hosts
            obs_rms_mids_all.append(np.sqrt(np.mean(obs['cal residual']**2)))
            obs_rms_errs_all.append(obs['cal residual'].std()/np.sqrt(len(obs['cal residual'])))

            model_rms_mids_all.append(np.sqrt(np.mean(g['mu_res']**2)))
            model_rms_errs_all.append(g['mu_res'].std()/np.sqrt(len(g['mu_res'])))
            model_c_mids_all.append(n.mid)
            obs_c_mids_all.append(n.mid)

            # High mass
            obshi = obs[obs['Host U-R']>colour_split]
            g1 = g[g['U-R']>colour_split]
            model_rms_mids_hi.append(np.sqrt(np.mean(g1['mu_res']**2)))
            model_rms_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))

            obs_rms_mids_hi.append(np.sqrt(np.mean(obshi['cal residual']**2)))
            obs_rms_errs_hi.append(obshi['cal residual'].std()/np.sqrt(len(obshi['cal residual'])))

            model_c_mids_hi.append(n.mid)
            obs_c_mids_hi.append(n.mid)


            # Low mass

            g2 = g[g['U-R']<=colour_split]
            model_rms_mids_lo.append(np.sqrt(np.mean(g2['mu_res']**2)))
            model_rms_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))

            obslo = obs[obs['Host U-R']<colour_split]
            obs_rms_mids_lo.append(np.sqrt(np.mean(obslo['cal residual']**2)))
            obs_rms_errs_lo.append(obslo['cal residual'].std()/np.sqrt(len(obslo['cal residual'])))

            model_c_mids_lo.append(n.mid)
            obs_c_mids_lo.append(n.mid)
        #except:
           #pass
    axURrms_all.plot(model_c_mids_all ,model_rms_mids_all,c='g',lw=3,label='Model All')
    axURrms_all.fill_between(model_c_mids_all ,np.array(model_rms_mids_all)-np.array(model_rms_errs_all),np.array(model_rms_mids_all)+np.array(model_rms_errs_all),color='g',lw=0.5,ls=':',alpha=0.3)
    axURrms_all.errorbar(obs_c_mids_all ,obs_rms_mids_all,yerr=obs_rms_errs_all,marker='D',color='k',linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR')


    axURrms.plot(model_c_mids_lo ,model_rms_mids_lo,c=split_colour_1,lw=3,label='Model Low Mass')
    axURrms.fill_between(model_c_mids_lo ,np.array(model_rms_mids_lo)-np.array(model_rms_errs_lo),np.array(model_rms_mids_lo)+np.array(model_rms_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)

    axURrms.plot(model_c_mids_hi ,model_rms_mids_hi,c=split_colour_2,lw=3,label='Model High Mass',ls='--')
    axURrms.fill_between(model_c_mids_hi ,np.array(model_rms_mids_hi)-np.array(model_rms_errs_hi),np.array(model_rms_mids_hi)+np.array(model_rms_errs_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)
    #axMASS.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')

    axURrms.errorbar(obs_c_mids_lo ,obs_rms_mids_lo,yerr=obs_rms_errs_lo,marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
    axURrms.errorbar(obs_c_mids_hi ,obs_rms_mids_hi,yerr=obs_rms_errs_hi,marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
    #chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_rms_mids_lo,model_c_mids_hi,model_rms_mids_hi)
    #axMASSrms.text(-0.2,-0.05,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
    plt.subplots_adjust(hspace=0,wspace=0)
    axURrms_all.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axURrms_all.tick_params(right=True,top=True,direction='in',labelsize=14)
    axURrms_all.set_ylim(0,0.4)


    axURrms_all.set_ylabel('RMS ($\mu_{\mathrm{res}}$)',size=20,)
    axURrms_all.legend(fontsize=13)
    axURrms.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axURrms.tick_params(right=True,top=True,direction='in',labelsize=14)
    axURrms.set_xlabel('$c$',size=20)
    axURrms.set_ylabel('RMS ($\mu_{\mathrm{res}}$)',size=20,)
    axURrms.legend(fontsize=13)
    axURrms_all.set_title(sim.save_string,size=20)
    axURrms.set_ylim(0,0.4)
    plt.savefig(sim.fig_dir +'HR_rms_c_split_UR_%s'%(sim.save_string)+label_ext)
