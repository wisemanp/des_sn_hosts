'''A set of routines for plotting AURA simulations'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_color_codes(palette='colorblind')
import itertools
import os
import pandas as pd
import pickle
from astropy.stats import poisson_conf_interval
from scipy.stats import halfnorm, skewnorm
from .HR_functions import calculate_step, get_red_chisq, get_red_chisq_interp, get_red_chisq_interp_splitx1


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
sim_colour= 'c'
data_colour = 'm'


def plot_galaxy_properties_paper(sim):

    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,4),sharey=True)
    cm =ax1.scatter(np.log10(sim.flux_df['mean_age']/1000),sim.flux_df['U']-sim.flux_df['R'],c=sim.flux_df['Av'],alpha=0.3)

    #ax1.set_xlabel('$\log (M_*/M_{\odot})$',size=20)
    ax1.set_xlabel('$\log($ Mean Stellar Age (Gyr))',size=20)
    ax1.set_ylabel('U-R',size=20)
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax1.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])


    cm =ax2.scatter(np.log10(sim.flux_df['ssfr']),sim.flux_df['U']-sim.flux_df['R'],c=sim.flux_df['Av'],alpha=0.3)

    ax2.set_xlabel('$\log (sSFR)$ yr$^{-1}$',size=20)
    #ax2.set_ylabel('U-R',size=20)
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax2.tick_params(which='both',labelsize=14,right=True,top=True)

    #ax3.set_xscale('log')
    ax3.set_xlabel('$\log (M_*/M_{\odot})$',size=20)
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    #ax3.set_ylabel('U-R',size=20)
    ax3.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
    cm =ax3.scatter(np.log10(sim.flux_df['mass']),sim.flux_df['U']-sim.flux_df['R'],c=sim.flux_df['Av'],alpha=0.3)
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
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    #ax3.set_ylabel('U-R',size=20)
    ax.tick_params(which='both',labelsize=14,right=True,top=True)
    #cbaxes = f.add_axes([0.2, 0.95, 0.6, 0.02])
    from matplotlib.colors import ListedColormap
    cm =ax.scatter(np.log10(sim.flux_df['mass']),sim.flux_df['U']-sim.flux_df['R'],c=sim.flux_df['Av'],alpha=0.3,
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
                linestyle='none',marker='+',label='DES U-R global',color='r')

    #ax.scatter(lisa_colours['Host Mass'],lisa_colours['Host U-R']+0.58,
    #
    #           marker='+',label='DES U-R global',c=lisa_colours['Host Mass'],cmap='gist_rainbow')
    ax.legend(loc='upper left',fontsize=15)
    ax.set_ylim(-0.5,3)
    plt.savefig(sim.fig_dir +'U-R_vs_data')


    f,ax=plt.subplots(figsize=(8,6.5))
    ax.hist(sim.sim_df['U-R'],density=True,bins=np.linspace(-0.5,2.5,100),histtype='step',lw=3,color=sim_colour,label='Sim')
    ax.hist(des5yr['Host U-R'],density=True,bins=np.linspace(-0.5,2.5,20),histtype='step',lw=3,color=data_colour,label='DES5YR')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.tick_params(right=True,top=True,which='both',labelsize=16)
    ax.set_xlabel('$U-R$',size=20)
    ax.set_ylabel('Normalized Frequency',size=20)
    ax.legend(fontsize=14)
    f,ax=plt.subplots(figsize=(8,6.5))

    ax.hist(np.log10(sim.sim_df['mass']),density=True,bins=np.linspace(7,12,100),histtype='step',lw=3,color=sim_colour,label='Sim')
    ax.hist(des5yr['Host Mass'],density=True,bins=np.linspace(7,12,20),histtype='step',lw=3,color=data_colour,label='DES5YR')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.tick_params(right=True,top=True,which='both',labelsize=16)
    ax.set_xlabel('$\log (M_*/M_{\odot})$',size=20)
    ax.set_ylabel('Normalized Frequency',size=20)
    ax.legend(fontsize=14)
def get_hist_errs(df,par,errext = '_err',axhist=False,linewidth=4.5,linestyle='-',lim_dist='skewnorm',label=None,bins=None,n=100,**kwargs):
    df['detection'] =True
    sorted_vals = df[par].sort_values()
    sorted_vals = sorted_vals[pd.notna(sorted_vals)]
    n=n

    adjusted_df = pd.DataFrame()
    if type(errext)==str:
        errcols =[par+errext]
    elif type(errext)==list:
        errcols = [errext[0],errext[1]]
    count_arr = []
    for i in range(n):
        detections = df[df['detection']==True]
        limits = df[df['detection']==False]
        if len(errcols)==1:
            res_adjusted = np.random.normal(detections[par].values,np.abs(detections[errcols[0]].values))
        if len(errcols) ==2:

            a = halfnorm(loc = np.zeros_like(detections[errcols[1]].values),
                               scale = np.abs(detections[errcols[1]].values-detections[par].values)).rvs(size=len(detections))
            b = -1*halfnorm(loc = np.zeros_like(detections[errcols[0]].values),
                               scale = np.abs(detections[par].values-detections[errcols[0]].values)).rvs(size=len(detections))

            sample = a+b
            res_adjusted = detections[par].values+sample
        if lim_dist =='skewnorm':
            res_lim_adjusted = skewnorm.rvs(-7,loc=limits[par].values,
                                                  scale=0.25,size=limits[par].values.size)
        elif lim_dist =='uniform':
            res_lim_adjusted = np.random.uniform(low=np.zeros_like(limits[par].values),
                                                 high=limits[par].values)
        elif lim_dist =='exp':
            res_lim_adjusted = limits[par].values - np.random.exponential(0.5,size=limits[par].values.size)
        elif lim_dist =='norm':
            res_lim_adjusted = np.random.normal(limits[par].values,
                                                np.ones_like(limits[par].values)*detections[errcol].mean())

        res_adjusted = np.concatenate([res_adjusted,res_lim_adjusted])
        sorted_res = np.sort(res_adjusted)
        sorted_res = sorted_res[np.nonzero(sorted_res)[0]]
        adjusted_df[i] = ''
        adjusted_df[i] = sorted_res
        #print(sorted_res)
        simcounts,simbins = np.histogram(sorted_res,density=False,bins=bins)
        count_arr.append(np.array(simcounts))

    bin_centers = (simbins [:-1] + simbins [1:])/2

    means,stds = [],[]
    count_arr = np.array(count_arr)
    for i in range(len(bin_centers)):
        means.append(np.mean(count_arr[:,i]))
        stds.append(np.std(count_arr[:,i]))
    return np.array(bin_centers),np.array(means),np.array(stds)
def plot_sample_hists(sim,label_ext='',):
    df = sim.sim_df
    f,(axc,axx1)=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
    df['detections'] =True
    bin_centers,means,stds = get_hist_errs(des5yr,'c',errext='ERR',n=100,bins=np.linspace(-0.3,0.3,20))

    # First do the histogram for plotting
    simcounts,simbins = np.histogram(sim.sim_df['c'],density=False,bins=np.linspace(-0.3,0.3,50))
    sim_bins = (simbins[:-1] + simbins[1:])/2
    simcounts = simcounts * len(des5yr)/len(sim.sim_df) * (bin_centers[-1]-bin_centers[-2])/(simbins[-1]-simbins[-2])
    # Now rebin the histogram to calculate the chi_squared
    simcounts_chi2,simbins_chi2 = np.histogram(sim.sim_df['c'],density=False,bins=np.linspace(-0.3,0.3,20))
    sim_bins_chi2 = (simbins_chi2[:-1] + simbins_chi2[1:])/2
    simcounts_chi2 = np.array(simcounts_chi2 * len(des5yr)/len(sim.sim_df) * (bin_centers[-1]-bin_centers[-2])/(simbins_chi2[-1]-simbins_chi2[-2]))

    chi2c = get_red_chisq(means,simcounts_chi2,stds)
    axc.step(sim_bins,simcounts,where='mid',color='c',label='Simulation',lw=3)
    axc.scatter(bin_centers,means,color='m',label='DES5YR',edgecolor='k',linewidth=0.8,zorder=6,s=50)
    axc.errorbar(bin_centers,means,yerr=stds,marker=None,linestyle='none',color='m',zorder=5)
    #ax.hist(pantheon['c'],density=True,bins=25,histtype='step',color='y',label='Obs Pantheon',lw=3)
    axc.legend(fontsize=15)
    axc.set_ylabel('N SNe',size=20)
    axc.set_xlabel('c',size=20)
    axc.text(0.12,70,r'$\chi^2_{\nu}=%.2f$'%chi2c,size=20)
    axc.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    axc.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axc.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axc.tick_params(which='both',direction='in',top=True,right=True,labelsize=16)

    df['logmass'] = np.log10(df['mass'])

    bin_centers,means,stds = get_hist_errs(des5yr,'x1',errext='ERR',n=100,bins=np.linspace(-3,3,20))

    simcounts,simbins = np.histogram(sim.sim_df['x1'],density=False,bins=np.linspace(-3,3,50))
    sim_bins = (simbins[:-1] + simbins[1:])/2
    simcounts = simcounts * len(des5yr)/len(sim.sim_df) * (bin_centers[-1]-bin_centers[-2])/(simbins[-1]-simbins[-2])

    simcounts_chi2,simbins_chi2 = np.histogram(sim.sim_df['x1'],density=False,bins=np.linspace(-3,3,20))
    sim_bins_chi2 = (simbins_chi2[:-1] + simbins_chi2[1:])/2
    simcounts_chi2 = simcounts_chi2 * len(des5yr)/len(sim.sim_df) * (bin_centers[-1]-bin_centers[-2])/(simbins_chi2[-1]-simbins_chi2[-2])

    chi2x1 = get_red_chisq(means,simcounts_chi2,stds)
    axx1.step(sim_bins,simcounts,where='mid',color='c',lw=3)
    axx1.scatter(bin_centers,means,color='m',edgecolor='k',linewidth=0.8,zorder=6,s=50)
    axx1.errorbar(bin_centers,means,yerr=stds,marker=None,linestyle='none',color='m',zorder=5)
    axx1.set_xlabel('$x_1$',size=20)
    axx1.text(1.2,70,r'$\chi^2_{\nu}=%.2f$'%chi2x1,size=20)
    #ax.hist(pantheon['x1'],density=True,bins=np.linspace(-3,3,20),histtype='step',lw=3,label='Pantheon')
    #axx1.legend()
    axx1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axx1.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axx1.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0,)
    axx1.tick_params(which='both',direction='in',top=True,right=True,labelsize=16)
    plt.savefig(sim.fig_dir +'SN_samples_%s'%(sim.save_string + '_paper')+label_ext)
    plt.savefig(sim.fig_dir +'SN_samples_%s'%(sim.save_string + '_paper')+label_ext+'.pdf')
    return chi2x1,chi2c

def plot_mu_res_paper(sim,obs=True,label_ext='',colour_split=1,mass_split=1E+10,return_chi=True):
    '''f,ax=plt.subplots(figsize=(8,6.5))
    ax.set_title(sim.save_string + '_paper',size=20)
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
    #ax.set_title(sim.save_string + '_paper',size=20)
    ax.hist(sim.sim_df['mu_res'],bins=100)
    ax.set_xlabel('$\mu_{\mathrm{res}}$',size=20)

    plt.savefig(sim.fig_dir +'HR_hist_%s'%(sim.save_string + '_paper')+label_ext)
    '''

    f,(axMASS,axUR)=plt.subplots(1,2,figsize=(10,6),sharey=True)
    #ax.set_title(sim.save_string + '_paper',size=20)
    cb=axUR.scatter(sim.sim_df['U-R'],sim.sim_df['mu_res'],alpha=0.3,c=sim_colour,label='Simulated data')#sim.sim_df['c'],cmap='rainbow')
    #plt.colorbar(cb,orientation='horizontal')
    counter=0
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,30))):
        if len(g)>0:
            label=None
            if counter==0:
                label='Binned simulation'
            else:
                label=None
            counter+=1
            axUR.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='b',marker='D',s=100,label=label)
            axUR.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='b',marker=None,ls='none')
    counter=0
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,3))):
        print(n.mid)
        label=None
        if counter==0:
            label='Sim split at step'
        else:
            label=None
        counter+=1
        print(g['mu_res'].std()/np.sqrt(len(g['mu_res'])))
        axUR.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),facecolor='none',
                      edgecolor='purple',linewidth=4,marker='s',s=100,label=label)
    leg =axUR.legend(loc='upper right',fontsize=15)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['U-R'],1)
    axUR.text(0.3,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=axUR.transAxes)
    axUR.set_xlabel('$U-R$',size=20)
    #axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    #ax.set_title(sim.save_string + '_paper',size=20)
    axUR.set_ylim(-0.3,0.3)
    axUR.set_xlim(-0.5,2.5)

    plt.savefig('test')
    l= axUR.get_xticklabels()

    l[0]=matplotlib.text.Text(-0.5,0,' ')
    axUR.set_xticklabels(l)
    axUR.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axUR.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axUR.tick_params(right=True,top=True,which='both',labelsize=16)
    sim.sim_df['logmass'] = np.log10(sim.sim_df['mass'])
    cb=axMASS.scatter(sim.sim_df['logmass'],sim.sim_df['mu_res'],alpha=0.3,c=sim_colour)#c=sim.sim_df['c'],cmap='rainbow')
    #plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,30))):
        if len(g)>0:
            axMASS.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='b',marker='D',s=100)
            axMASS.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='b',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,3))):
        print(n.mid)
        axMASS.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),ecolor='b',markerfacecolor='none',markeredgecolor='purple',mew=4,marker='s',markersize=20,)
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['logmass'],10)
    axMASS.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=axMASS.transAxes)
    axMASS.set_xlabel('$\log(M_*/M_{\odot})$',size=20)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axMASS.set_xlim(7.5,12)
    axMASS.set_ylim(-0.3,0.3)
    axMASS.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    axMASS.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axMASS.tick_params(right=True,top=True,which='both',labelsize=16)
    #ax.set_title(sim.save_string + '_paper',size=20)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)

    plt.savefig(sim.fig_dir +'HR_vs_host_scatter_%s'%(sim.save_string + '_paper')+label_ext)
    '''f,ax=plt.subplots(figsize=(8,6.5))
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
    #ax.set_title(sim.save_string + '_paper',size=20)
    ax.set_ylim(-0.3,0.3)

    plt.savefig(sim.fig_dir +'/HR_vs_age_scatter_%s'%(sim.save_string + '_paper')+label_ext)'''
    chis = []
    fMASSUR,(axMASS,axUR)=plt.subplots(1,2,figsize=(10,6),sharey=True)
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
    axMASS.set_xlabel('$c$',size=20)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axMASS.legend(fontsize=13)
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

    if obs:
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
    axUR.set_xlabel('$c$',size=20)
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

    plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_%s'%(sim.save_string + '_paper')+label_ext)

    return chis
def plot_rms_paper(sim,label_ext='',colour_split=1,mass_split=1E+10):

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
    axMASSrms_all.set_title(sim.save_string + '_paper',size=20)
    axMASSrms.set_ylim(0,0.4)
    plt.savefig(sim.fig_dir +'HR_rms_c_split_mass_%s'%(sim.save_string + '_paper')+label_ext)

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
    axURrms_all.set_title(sim.save_string + '_paper',size=20)
    axURrms.set_ylim(0,0.4)
    plt.savefig(sim.fig_dir +'HR_rms_c_split_UR_%s'%(sim.save_string + '_paper')+label_ext)

def plot_mu_res_paper_splitx1(sim,obs=True,label_ext='',colour_split=1,mass_split=1E+10,return_chi=True):
    '''f,ax=plt.subplots(figsize=(8,6.5))
    ax.set_title(sim.save_string + '_paper',size=20)
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
    #ax.set_title(sim.save_string + '_paper',size=20)
    ax.hist(sim.sim_df['mu_res'],bins=100)
    ax.set_xlabel('$\mu_{\mathrm{res}}$',size=20)
    plt.savefig(sim.fig_dir +'HR_hist_%s'%(sim.save_string + '_paper')+label_ext)
    '''

    f,(axMASS,axUR)=plt.subplots(1,2,figsize=(12,6),sharey=True)
    #ax.set_title(sim.save_string + '_paper',size=20)
    cb=axUR.scatter(sim.sim_df['U-R'],sim.sim_df['mu_res'],alpha=0.3,c=sim_colour,label='Simulated data')#sim.sim_df['c'],cmap='rainbow')
    #plt.colorbar(cb,orientation='horizontal')
    counter=0
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,30))):
        if len(g)>0:
            label=None
            if counter==0:
                label='Binned simulation'
            else:
                label=None
            counter+=1
            axUR.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='b',marker='D',s=100,label=label)
            axUR.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='b',marker=None,ls='none')
    counter=0
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['U-R'],bins=np.linspace(-0.5,2.5,3))):
        print(n.mid)
        label=None
        if counter==0:
            label='Sim split at step'
        else:
            label=None
        counter+=1
        print(g['mu_res'].std()/np.sqrt(len(g['mu_res'])))
        axUR.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),facecolor='none',
                      edgecolor='purple',linewidth=4,marker='s',s=100,label=label)
    leg =axUR.legend(loc='upper right',fontsize=15)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['U-R'],1)
    axUR.text(0.3,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=axUR.transAxes)
    axUR.set_xlabel('$U-R$',size=20)
    #axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    #ax.set_title(sim.save_string + '_paper',size=20)
    axUR.set_ylim(-0.3,0.3)
    axUR.set_xlim(-0.5,2.5)

    plt.savefig('test')
    l= axUR.get_xticklabels()

    l[0]=matplotlib.text.Text(-0.5,0,' ')
    axUR.set_xticklabels(l)
    axUR.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axUR.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axUR.tick_params(right=True,top=True,which='both',labelsize=16)
    sim.sim_df['logmass'] = np.log10(sim.sim_df['mass'])
    cb=axMASS.scatter(sim.sim_df['logmass'],sim.sim_df['mu_res'],alpha=0.3,c=sim_colour)#c=sim.sim_df['c'],cmap='rainbow')
    #plt.colorbar(cb)
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,30))):
        if len(g)>0:
            axMASS.scatter(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),c='b',marker='D',s=100)
            axMASS.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),c='b',marker=None,ls='none')
    for n,g in sim.sim_df.groupby(pd.cut(sim.sim_df['logmass'],bins=np.linspace(8,12,3))):
        print(n.mid)
        axMASS.errorbar(n.mid,np.average(g['mu_res'],weights=(1/(g['mu_res_err'])**2)),yerr=g['mu_res'].std()/np.sqrt(len(g['mu_res'])),ecolor='b',markerfacecolor='none',markeredgecolor='purple',mew=4,marker='s',markersize=20,)
    step,sig = calculate_step(sim.sim_df['mu_res'], sim.sim_df['mB_err'],sim.sim_df['logmass'],10)
    axMASS.text(0.1,0.1,'%.3f mag, $%.2f \sigma$'%(step,sig),transform=axMASS.transAxes)
    axMASS.set_xlabel('$\log(M_*/M_{\odot})$',size=20)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axMASS.set_xlim(7.5,12)
    axMASS.set_ylim(-0.3,0.3)
    axMASS.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    axMASS.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axMASS.tick_params(right=True,top=True,which='both',labelsize=16)
    #ax.set_title(sim.save_string + '_paper',size=20)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)

    plt.savefig(sim.fig_dir +'HR_vs_host_scatter_%s'%(sim.save_string + '_paper')+label_ext)

    chis = []
    fMASSUR,(axMASS,axUR)=plt.subplots(1,2,figsize=(10,6),sharey=True)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi_hi , model_hr_mids_hi_hi ,  model_hr_errs_hi_hi, model_c_mids_hi_lo , model_hr_mids_hi_lo ,  model_hr_errs_hi_lo =[],[],[],[],[],[],[],[],[]

    x1_split=-0.3

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['mass']>mass_split]
            g1x1hi = g1[g1['x1']>x1_split]

            model_hr_mids_hi_hi.append(np.average(g1x1hi['mu_res'],weights=1/g1x1hi['mu_res_err']**2))
            model_hr_errs_hi_hi.append(g1x1hi['mu_res'].std()/np.sqrt(len(g1x1hi['mu_res'])))
            model_c_mids_hi_hi.append(n.mid)

            g1x1lo = g1[g1['x1']<=x1_split]

            model_hr_mids_hi_lo.append(np.average(g1x1lo['mu_res'],weights=1/g1x1lo['mu_res_err']**2))
            model_hr_errs_hi_lo.append(g1x1lo['mu_res'].std()/np.sqrt(len(g1x1lo['mu_res'])))
            model_c_mids_hi_lo.append(n.mid)


            g2 = g[g['mass']<=mass_split]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass

    axMASS.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Low Mass')
    axMASS.fill_between(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)

    axMASS.plot(model_c_mids_hi_hi ,model_hr_mids_hi_hi,c=split_colour_2,lw=3,label='Model High Mass High x1',ls='--')
    axMASS.fill_between(model_c_mids_hi_hi ,np.array(model_hr_mids_hi_hi)-np.array(model_hr_errs_hi_hi),np.array(model_hr_mids_hi_hi)+np.array(model_hr_errs_hi_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)

    axMASS.plot(model_c_mids_hi_lo ,model_hr_mids_hi_lo,c='m',lw=3,label='Model High Mass Low x1',ls=':')
    axMASS.fill_between(model_c_mids_hi_lo ,np.array(model_hr_mids_hi_lo)-np.array(model_hr_errs_hi_lo),np.array(model_hr_mids_hi_lo)+np.array(model_hr_errs_hi_lo),color='m',lw=0.5,ls=':',alpha=0.3)

    #axMASS.plot(model_c_mids_hi ,np.array(model_hr_mids_hi)-np.array(model_hr_errs_hi),c=split_colour_2,lw=0.5,ls=':')
    if obs:
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
        #chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        #axMASS.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        #chis.append(chisq)
    axMASS.set_xlabel('$c$',size=20)
    axMASS.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axMASS.legend(fontsize=10)
    #axMASS.set_title(sim.save_string + '_paper',size=20)
    axMASS.set_ylim(-0.2,0.2)
    axMASS.set_xlim(-0.19,0.3)
    axMASS.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    axMASS.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    axMASS.tick_params(which='both',right=True,top=True,labelsize=16)
    #plt.savefig(sim.fig_dir +'HR_vs_c_split_mass_%s'%(sim.save_string + '_paper')+label_ext)
    #fUR,axUR=plt.subplots(figsize=(8,6.5))
    #axUR.set_title(sim.save_string + '_paper',size=20)
    model_c_mids_lo , model_hr_mids_lo , model_hr_errs_lo , model_c_mids_hi_hi , model_hr_mids_hi_hi ,  model_hr_errs_hi_hi, model_c_mids_hi_lo , model_hr_mids_hi_lo ,  model_hr_errs_hi_lo =[],[],[],[],[],[],[],[],[]

    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):
        try:
            g1 = g[g['U-R']>colour_split]
            g1x1hi = g1[g1['x1']>x1_split]

            model_hr_mids_hi_hi.append(np.average(g1x1hi['mu_res'],weights=1/g1x1hi['mu_res_err']**2))
            model_hr_errs_hi_hi.append(g1x1hi['mu_res'].std()/np.sqrt(len(g1x1hi['mu_res'])))
            model_c_mids_hi_hi.append(n.mid)

            g1x1lo = g1[g1['x1']<=x1_split]

            model_hr_mids_hi_lo.append(np.average(g1x1lo['mu_res'],weights=1/g1x1lo['mu_res_err']**2))
            model_hr_errs_hi_lo.append(g1x1lo['mu_res'].std()/np.sqrt(len(g1x1lo['mu_res'])))
            model_c_mids_hi_lo.append(n.mid)
            g2 = g[g['U-R']<=colour_split]

            model_hr_mids_lo.append(np.average(g2['mu_res'],weights=1/g2['mu_res_err']**2))
            model_hr_errs_lo.append(g2['mu_res'].std()/np.sqrt(len(g2['mu_res'])))
            model_c_mids_lo.append(n.mid)
        except:
            pass

    axUR.plot(model_c_mids_lo ,model_hr_mids_lo,c=split_colour_1,lw=3,label='Model Blue Host')
    axUR.fill_between(model_c_mids_lo ,np.array(model_hr_mids_lo)-np.array(model_hr_errs_lo),np.array(model_hr_mids_lo)+np.array(model_hr_errs_lo),color=split_colour_1,lw=0.5,ls=':',alpha=0.3)
    axUR.plot(model_c_mids_hi_hi ,model_hr_mids_hi_hi,c=split_colour_2,lw=3,label='Model Red Host High x1',ls='--')
    axUR.fill_between(model_c_mids_hi_hi ,np.array(model_hr_mids_hi_hi)-np.array(model_hr_errs_hi_hi),np.array(model_hr_mids_hi_hi)+np.array(model_hr_errs_hi_hi),color=split_colour_2,lw=0.5,ls=':',alpha=0.3)

    axUR.plot(model_c_mids_hi_lo ,model_hr_mids_hi_lo,c='m',lw=3,label='Model Blue Host Low x1',ls=':')
    axUR.fill_between(model_c_mids_hi_lo ,np.array(model_hr_mids_hi_lo)-np.array(model_hr_errs_hi_lo),np.array(model_hr_mids_hi_lo)+np.array(model_hr_errs_hi_lo),color='m',lw=0.5,ls=':',alpha=0.3)

    if obs:
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
        #chisq =get_red_chisq_interp(low,high,model_c_mids_lo,model_hr_mids_lo,model_c_mids_hi,model_hr_mids_hi)
        axUR.errorbar(low['c'],low['hr'],xerr=low['c_err'],yerr =low['hr_err'],marker='D',color=split_colour_1,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R<1$')
        axUR.errorbar(high['c'],high['hr'],xerr=high['c_err'],yerr =high['hr_err'],marker='D',color=split_colour_2,linestyle='none',markersize=10,alpha=0.8,mew=1.5,mec='w',label='DES5YR global $U-R>1$')
        #axUR.text(0.,0.15,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
        #chis.append(chisq)
    axUR.set_xlabel('$c$',size=20)
    #axUR.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    axUR.legend(fontsize=10,loc='lower left')
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

    plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_%s'%(sim.save_string + '_paper')+label_ext)

    return chis

def plot_mu_res_paper_splitssfr(sim,obs=True,label_ext='',colour_split=1,mass_split=1E+10,return_chi=True):
    mvdf = pd.read_csv('/media/data3/wiseman/des/AURA/data/data_nozcut_snnv19.csv')

    oldest,oldish,youngish,youngest = ['#FF9100','red','purple','darkblue']

    f,ax=plt.subplots(figsize=(8,6.5),sharey=True)
    model_c_mids_lo_lo , model_hr_mids_lo_lo , model_hr_errs_lo_lo ,model_c_mids_mid_lo , model_hr_mids_mid_lo , model_hr_errs_mid_lo = [],[],[],[],[],[]
    model_c_mids_mid_hi , model_hr_mids_mid_hi , model_hr_errs_mid_hi, model_c_mids_hi_hi , model_hr_mids_hi_hi ,  model_hr_errs_hi_hi = [],[],[],[],[],[]
    x1_split=-0.3
    ssfr_lo_split = -10.5
    ssfr_hi_split = -9.5
    sim.sim_df['log_ssfr'] = np.log10(sim.sim_df['ssfr'])
    for counter,(n,g) in enumerate(sim.sim_df.groupby(pd.cut(sim.sim_df['c'],bins=np.linspace(-0.3,0.3,20)))):

        g1 = g[g['log_ssfr']>ssfr_hi_split]
        g1x1hi = g1[g1['x1']>x1_split]
        if len(g1x1hi)>0:
            model_c_mids_hi_hi.append(n.mid)
            model_hr_mids_hi_hi.append(np.average(g1x1hi['mu_res'],weights=1/g1x1hi['mu_res_err']**2))
            model_hr_errs_hi_hi.append(g1x1hi['mu_res'].std()/np.sqrt(len(g1x1hi['mu_res'])))

        g2 = g[(g['log_ssfr']<=ssfr_hi_split)&(g['ssfr']>ssfr_lo_split)]
        g2x1hi = g2[g2['x1']>x1_split]
        if len(g2x1hi)>0:

            model_c_mids_mid_hi.append(n.mid)
            model_hr_mids_mid_hi.append(np.average(g2x1hi['mu_res'],weights=1/g2x1hi['mu_res_err']**2))
            model_hr_errs_mid_hi.append(g2x1hi['mu_res'].std()/np.sqrt(len(g2x1hi['mu_res'])))
        g2x1lo = g2[g2['x1']<=x1_split]
        if len(g2x1lo)>0:
            model_c_mids_mid_lo.append(n.mid)
            model_hr_mids_mid_lo.append(np.average(g2x1lo['mu_res'],weights=1/g2x1lo['mu_res_err']**2))
            model_hr_errs_mid_lo.append(g2x1lo['mu_res'].std()/np.sqrt(len(g2x1lo['mu_res'])))
        g3 = g[g['log_ssfr']<=ssfr_lo_split]
        g3x1lo = g3[g3['x1']<x1_split]
        if len(g3x1lo)>0:
            model_c_mids_lo_lo.append(n.mid)
            model_hr_mids_lo_lo.append(np.average(g3x1lo['mu_res'],weights=1/g3x1lo['mu_res_err']**2))
            model_hr_errs_lo_lo.append(g3x1lo['mu_res'].std()/np.sqrt(len(g3x1lo['mu_res'])))



    ax.plot(model_c_mids_lo_lo ,model_hr_mids_lo_lo,c=split_colour_1,lw=3,label='Model low sSFR; low $x_1$',color=oldest)
    ax.fill_between(model_c_mids_lo_lo,np.array(model_hr_mids_lo_lo)-np.array(model_hr_errs_lo_lo),np.array(model_hr_mids_lo_lo)+np.array(model_hr_errs_lo_lo),color=oldest,lw=0.5,ls=':',alpha=0.1)

    ax.plot(model_c_mids_mid_lo ,model_hr_mids_mid_lo,c='m',lw=3,label='Model mid sSFR; low $x_1$',ls=':',color=oldish)
    ax.fill_between(model_c_mids_mid_lo ,np.array(model_hr_mids_mid_lo)-np.array(model_hr_errs_mid_lo),np.array(model_hr_mids_mid_lo)+np.array(model_hr_errs_mid_lo),color=oldish,lw=0.5,ls=':',alpha=0.1)

    ax.plot(model_c_mids_mid_hi ,model_hr_mids_mid_hi,c='m',lw=3,label='Model mid sSFR; high $x_1$',ls='-.',color=youngish)
    ax.fill_between(model_c_mids_mid_hi ,np.array(model_hr_mids_mid_hi)-np.array(model_hr_errs_mid_hi),np.array(model_hr_mids_mid_hi)+np.array(model_hr_errs_mid_hi),color=youngish,lw=0.5,ls=':',alpha=0.1)


    ax.plot(model_c_mids_hi_hi ,model_hr_mids_hi_hi,c=split_colour_2,lw=3,label='Model high sSFR; high $x_1$',ls='--',color=youngest)
    ax.fill_between(model_c_mids_hi_hi ,np.array(model_hr_mids_hi_hi)-np.array(model_hr_errs_hi_hi),np.array(model_hr_mids_hi_hi)+np.array(model_hr_errs_hi_hi),color=youngest,lw=0.5,ls=':',alpha=0.1)

    ax.errorbar(mvdf['avg_colour_log(sSFR)<-10.6,x_1<-0.3'],mvdf['avg_mures_log(sSFR)<-10.6,x_1<-0.3'],
        yerr=mvdf['stdm_mures_log(sSFR)<-10.6,x_1<-0.3'],marker='D',
        color=oldest,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data low sSFR; low $x_1$')
    ax.errorbar(mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1<-0.3'],mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'],
        yerr=mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1<-0.3'],marker='o',
        color=oldish,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data mid sSFR; low $x_1$')

    ax.errorbar(mvdf['avg_colour_-10.6<log(sSFR)<-9.5,x_1>-0.3'],mvdf['avg_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'],
        yerr=mvdf['stdm_mures_-10.6<log(sSFR)<-9.5,x_1>-0.3'],marker='s',
        color=youngish,linestyle='none',markersize=10,mec='k',mew=0.5,label='Data mid sSFR; high $x_1$')

    ax.errorbar(mvdf['avg_colour_log(sSFR)<-9.5,x_1>-0.3'],mvdf['avg_mures_log(sSFR)<-9.5,x_1>-0.3'],
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
    ax.text(-0.1,-0.25,r'$\chi^2_{\nu}=%.2f$'%chisq,size=20)
    ax.set_xlabel('$c$',size=20)
    ax.set_ylabel('$\mu_{\mathrm{res}}$',size=20,)
    ax.legend(fontsize=10,ncol=2,loc='upper center')
    #axMASS.set_title(sim.save_string + '_paper',size=20)
    ax.set_ylim(-0.3,0.3)
    ax.set_xlim(-0.19,0.3)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(which='both',right=True,top=True,labelsize=16)

    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(sim.fig_dir +'HR_vs_c_split_sSFR_x1_%s'%(sim.save_string + '_paper')+label_ext)

    return chisq
