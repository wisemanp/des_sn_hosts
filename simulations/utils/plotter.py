'''A set of routines for plotting AURA simulations'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes(palette='colorblind')
import itertools
import os
import pandas as pd
from .HR_functions import calculate_step
des5yr = pd.read_csv('/media/data3/wiseman/des/AURA/data/df_after_cuts_z0.6_UR1.csv')
aura_dir = os.environ['AURA_DIR']

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
    lisa_colours = pd.read_csv(aura_dir+'5yr-massUR.csv',index_col=0)

    ax.errorbar(lisa_colours['Host Mass'],lisa_colours['Host U-R'],
                xerr=lisa_colours['Host Mass error'],yerr=lisa_colours['Host U-R error'],
                linestyle='none',marker='+',label='DES U-R global')

    #ax.scatter(lisa_colours['Host Mass'],lisa_colours['Host U-R']+0.58,
    #
    #           marker='+',label='DES U-R global',c=lisa_colours['Host Mass'],cmap='gist_rainbow')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim(-0.5,3)
    plt.savefig(sim.fig_dir +'U-R_vs_data')


def plot_x1s(sim,df):
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,5),sharey=True)
    ax1.scatter(df['mass'],df['x1'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax1.set_xscale('log')
    cm=ax2.scatter(df['U-R'],df['x1'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax3.scatter(df['SN_age'],df['x1'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax3.set_xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='vertical',ax=ax3)#shrink=0.7)
    ax1.set_ylabel('$x_1$',size=20)
    cb.set_label('$A_V$',size=20,
                )
    for ax in [ax1,ax2]:
        ax.tick_params(which='both',direction='in',top=True,right=True)
    ax1.set_xlabel('Stellar Mass',size=20)
    ax2.set_xlabel('$U-R$',size=20)
    ax3.set_xlabel('SN age (Gyr)',size=20)
    ax2.set_xlim(0,2.5)

    f,ax=plt.subplots(figsize=(8,6.5))
    hist=ax.hist(df['x1'],bins=np.linspace(-3,3,100),density=True,label='Simulation',histtype='step',lw=3)
    ax.set_xlabel('$x_1$',size=20)
    ax.hist(des5yr['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='DES 5yr')
    #ax.hist(pantheon['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='Pantheon')
    ax.legend()
    plt.savefig(sim.fig_dir +'SN_x1_hist_%s'%sim.save_string)
def plot_cs(sim,df):
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,5),sharey=True)
    ax1.scatter(df['mass'],df['c'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax1.set_xscale('log')
    cm=ax2.scatter(df['U-R'],df['c'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax3.scatter(df['SN_age'],df['c'],c=df['host_Av'],alpha=0.6,edgecolor='w',lw=0.1,cmap='viridis')
    ax3.set_xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    cb=plt.colorbar(cm,orientation='vertical',ax=ax3)#shrink=0.7)
    ax1.set_ylabel('$c$',size=20)
    cb.set_label('$A_V$',size=20)
    for ax in [ax1,ax2]:
        ax.tick_params(which='both',direction='in',top=True,right=True)
    ax1.set_xlabel('Stellar Mass',size=20)
    ax2.set_xlabel('$U-R$',size=20)
    ax3.set_xlabel('SN age (Gyr)',size=20)
    ax2.set_xlim(0,2)
    ax1.set_ylim(-0.32,0.4)
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.hist(df['c'],bins=np.linspace(-0.3,0.3,25),histtype='step',density=True,label='Sim',lw=3)
    ax.hist(des5yr['c'],density=True,bins=25,histtype='step',color=split_colour_1,label='Obs DES',lw=3)
    #ax.hist(pantheon['c'],density=True,bins=25,histtype='step',color='y',label='Obs Pantheon',lw=3)
    ax.legend()
    ax.set_xlabel('c',size=20)
    plt.savefig(sim.fig_dir +'SN_c_hist_%s'%sim.save_string)
def plot_hosts(sim,df):
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.scatter(df['mass'],df['host_Av'],alpha=0.1,c='c',edgecolor='w')
    ax.set_xscale('log')
    ax.set_xlabel('Stellar Mass',size=20)
    ax.set_ylabel('$A_V$',size=20)
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.scatter(df['mass'],df['Rv'],alpha=0.1,c='c',edgecolor='w')
    ax.set_xscale('log')
    ax.set_xlabel('Stellar Mass',size=20)
    ax.set_ylabel('$R_V$',size=20)
    ax.set_ylim(1,6)
    f,ax=plt.subplots(figsize=(8,6.5))
    ax.scatter(df['host_Av'],df['x1'],alpha=0.1,c='c',edgecolor='w')

    ax.set_ylabel('$x_1$',size=20)
    ax.set_xlabel('$A_V$',size=20)
def plot_samples(sim,zmin=0,zmax=1.2,x1=True,c=True,hosts=True):
    plot_df=sim.sim_df[(sim.sim_df['z']>zmin)&(sim.sim_df['z']<zmax)]
    if x1:
        sim.plot_x1s(plot_df)
    if c:
        sim.plot_cs(plot_df)
    if hosts:
        sim.plot_hosts(plot_df)
def plot_mu_res(sim,obs=True,label_ext=''):
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
        ax.scatter(n.mid,g['mu_res'].median(),c='r',marker='D',s=100)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,3))):
        print(n.mid)
        ax.errorbar(n.mid,g['mu_res'].mean(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='c',marker='s',markersize=20,ls='none')

    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['U-R'],1)
    ax.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=ax.transAxes)
    ax.set_xlabel('$U-R$',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.set_title(sim.save_string,size=20)
    ax.set_ylim(-0.3,0.3)
    ax.set_xlim(0,2.5)

    plt.savefig(sim.fig_dir +'HR_vs_UR_scatter_%s'%(sim.save_string)+label_ext)
    f,ax=plt.subplots(figsize=(8,6.5))
    sim.sim_df['logmass'] = np.log10(sim.sim_df['mass'])
    cb=ax.scatter(sim.sim_df['logmass'],sim.sim_df['mu_res'],alpha=0.3,c=sim.sim_df['c'],cmap='rainbow')
    plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,30))):
        ax.scatter(n.mid,g['mu_res'].median(),c='r',marker='D',s=100)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
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
        ax.scatter(n.mid,g['mu_res'].median(),c='r',marker='D',s=100)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='r',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logSN_age'],bins=np.linspace(-1.5,1,3))):
        print(n.mid)
        ax.errorbar(n.mid,g['mu_res'].median(),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='c',marker='s',markersize=20,ls='none')
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mb_err'],sim.sim_df['logSN_age'],0.75)
    ax.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=ax.transAxes)
    ax.set_xlabel('$\log$ (SN Age (Gyr))',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.set_title(sim.save_string,size=20)
    ax.set_ylim(-0.3,0.3)

    plt.savefig(sim.fig_dir +'/HR_vs_age_scatter_%s'%(sim.save_string)+label_ext)
    fMASS,axMASS=plt.subplots(figsize=(8,6.5))
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi , model_hr_mids_hi ,  model_hr_errs_hi =[],[],[],[],[],[]
    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['mass']>1E+10]


            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['mass']<=1E+10]

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
            g1 = g[g['U-R']>1.]

            model_hr_mids_hi.append(np.average(g1['mu_res'],weights=1/g1['mu_res_err']**2))
            model_hr_errs_hi.append(g1['mu_res'].std()/np.sqrt(len(g1['mu_res'])))
            model_c_mids_hi.append(n.mid)
            g2 = g[g['U-R']<=1.]

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
        axUR.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr =low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
        axUR.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr =high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
    axUR.set_xlabel('$c$',size=20)
    axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axUR.legend(fontsize=13)
    axUR.set_ylim(-0.2,0.2)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_UR_%s'%(sim.save_string)+label_ext)
