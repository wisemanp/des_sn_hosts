'''Galaxy-related functions'''
import numpy as np
import pandas as pd
def double_schechter(logM,logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):#logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):
    if phi_star_1<0:
        phi_star_1 = 10**phi_star_1
    if phi_star_2<0:
        phi_star_2 = 10**phi_star_2
    return np.log(10)*((phi_star_1*(10**((logM - logM_star)*(1+alpha_1))))+(phi_star_2*(10**((logM - logM_star)*(1+alpha_2)))))*\
                np.exp(-10**(logM-logM_star))


def double_schechter_T14(logM,logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):#logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):
    ''' Tomczak et al 2014 implementation of the double schechter function'''
    if phi_star_1<0:
        phi_star_1 = 10**phi_star_1
    if phi_star_2<0:
        phi_star_2 = 10**phi_star_2
    return np.log(10)*np.exp(-10**(logM-logM_star))*10**(logM-logM_star)*\
    ((phi_star_1*(10**((logM - logM_star)*(alpha_1))))+(phi_star_2*(10**((logM - logM_star)*(alpha_2)))))


def single_schechter(logM,logM_star,phi_star,alpha,):
    return np.log(10)*((phi_star*(10**((logM - logM_star)*(1+alpha))))*\
                np.exp(-10**(logM-logM_star)))

zfourge = {
    0.2: {'logM_star':10.78,'phi_star_1':-2.54,'alpha_1':-0.98,'phi_star_2':-4.29,'alpha_2':-1.9},
    0.5: {'logM_star':10.7,'phi_star_1':-2.55,'alpha_1':-0.39,'phi_star_2':-3.15,'alpha_2':-1.53},
    0.75: {'logM_star':10.66,'phi_star_1':-2.56,'alpha_1':-0.37,'phi_star_2':-3.49,'alpha_2':-1.62},
    1: {'logM_star':10.54,'phi_star_1':-2.72,'alpha_1':0.3,'phi_star_2':-3.17,'alpha_2':-1.45},
}

def schechter(z,logM):
    if z<0.5:

        return double_schechter_T14(logM,**zfourge[0.2])
    elif z>=0.5 and z<0.75:
        return double_schechter_T14(logM,**zfourge[0.5])
    elif z>=0.75 and z<1:
        return double_schechter_T14(logM,**zfourge[0.75])
    elif z>=1 and z<1.25:
        return double_schechter_T14(logM,**zfourge[1])

def ozdes_efficiency():
    import numpy.polynomial.polynomial as poly
    from scipy.interpolate import interp1d
    fs = ['C12','X12','S12','E12','C3','X3']
    ys = ['123','4','5']
    efficiencies = {}
    for f in fs:
        for y in ys:
            eff = pd.read_csv('/media/data3/wiseman/des/desdtd/efficiencies/eff_%s_Y%s.dat'%(f,y),sep=' ',skipinitialspace=True)
            coefs = poly.polyfit(eff['r_obs'],eff['HOSTEFF'], 13)
            slope_start = eff['r_obs'].loc[eff.sort_values('r_obs',ascending=False)['HOSTEFF'].idxmax()]
            slope_end = eff['r_obs'].loc[eff['HOSTEFF'].idxmin()]
            x_new=np.linspace(slope_start,slope_end,1000)
            ffit = poly.polyval(x_new, coefs)
            x_new = np.concatenate([np.linspace(13,slope_start,50),x_new])
            ffit = np.concatenate([np.ones_like(np.linspace(13,slope_start,50)),ffit])
            x_new = np.concatenate([x_new,np.linspace(slope_end,32,50)])
            ffit = np.concatenate([ffit,np.zeros_like(np.linspace(slope_end,32,50))])
            completeness_func = interp1d(x_new,ffit)
            efficiencies[f+'_Y'+y] = completeness_func

    mags = np.linspace(13,32,100)
    eff_df = pd.DataFrame(index=mags)
    for y in ['Y123','Y4','Y5']:
        for f in ['X12','C12','S12','E12','X3','C3']:
            eff = np.clip(efficiencies['%s_%s'%(f,y)](mags),a_min=0,a_max=None)
            eff_df['%s_%s'%(f,y)] = eff
    mean_eff_func = interp1d(mags,eff_df.mean(axis=1))
    std_eff_func =  interp1d(mags,eff_df.std(axis=1))

    return mean_eff_func,std_eff_func
