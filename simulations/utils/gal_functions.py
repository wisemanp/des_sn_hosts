'''Galaxy-related functions'''
import numpy as np
import pandas as pd
import os
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
    0:{  #Star forming
    0.2: {'logM_star':10.59,'phi_star_1':-2.67,'alpha_1':-1.08,'phi_star_2':-4.46,'alpha_2':-2.0},
    0.5: {'logM_star':10.765,'phi_star_1':-2.97,'alpha_1':-0.97,'phi_star_2':-3.34,'alpha_2':-1.58},
    0.75: {'logM_star':10.56,'phi_star_1':-2.81,'alpha_1':-0.46,'phi_star_2':-3.36,'alpha_2':-1.61},
    1: {'logM_star':10.44,'phi_star_1':-2.98,'alpha_1':0.53,'phi_star_2':-3.12,'alpha_2':-1.44},
    },
    1:{  # Quench
    0.2: {'logM_star':10.75,'phi_star_1':-2.76,'alpha_1':-0.47,'phi_star_2':-5.21,'alpha_2':-1.97},
    0.5: {'logM_star':10.68,'phi_star_1':-2.67,'alpha_1':-0.10,'phi_star_2':-4.29,'alpha_2':-1.69},
    0.75: {'logM_star':10.63,'phi_star_1':-2.81,'alpha_1':0.04,'phi_star_2':-4.40,'alpha_2':-1.51},
    1: {'logM_star':10.63,'phi_star_1':-3.03,'alpha_1':0.11,'phi_star_2':-4.80,'alpha_2':-1.57},
    }
}

def schechter(z,logM,SF):
    if z<0.5:
        return double_schechter_T14(logM,**zfourge[SF][0.2])
    elif z>=0.5 and z<0.75:
        return double_schechter_T14(logM,**zfourge[SF][0.5])
    elif z>=0.75 and z<1:
        return double_schechter_T14(logM,**zfourge[SF][0.75])
    elif z>=1 and z<1.25:
        return double_schechter_T14(logM,**zfourge[SF][1])

def ozdes_efficiency(dir='/media/data3/wiseman/des/desdtd/efficiencies/'):
    import numpy.polynomial.polynomial as poly
    from scipy.interpolate import interp1d
    fs = ['C12','X12','S12','E12','C3','X3']
    ys = ['123','4','5']
    efficiencies = {}
    for f in fs:
        for y in ys:
            eff = pd.read_csv(os.path.join(dir,'efficiencies/eff_%s_Y%s.dat'%(f,y)),sep=' ',skipinitialspace=True)
            coefs = poly.polyfit(eff['r_obs'],eff['HOSTEFF'], 13)
            slope_start = eff['r_obs'].loc[eff.sort_values('r_obs',ascending=False)['HOSTEFF'].idxmax()]
            slope_end = eff['r_obs'].loc[eff['HOSTEFF'].idxmin()]
            x_new=np.linspace(slope_start,slope_end,1000)
            ffit = poly.polyval(x_new, coefs)
            x_new = np.concatenate([np.linspace(10,slope_start,50),x_new])
            ffit = np.concatenate([np.ones_like(np.linspace(10,slope_start,50)),ffit])
            x_new = np.concatenate([x_new,np.linspace(slope_end,39,50)])
            ffit = np.concatenate([ffit,np.zeros_like(np.linspace(slope_end,39,50))])
            completeness_func = interp1d(x_new,ffit)
            efficiencies[f+'_Y'+y] = completeness_func

    mags = np.linspace(10,39,500)
    eff_df = pd.DataFrame(index=mags)
    for y in ['Y123','Y4','Y5']:
        for f in ['X12','C12','S12','E12','X3','C3']:
            eff = np.clip(efficiencies['%s_%s'%(f,y)](mags),a_min=0,a_max=None)
            eff_df['%s_%s'%(f,y)] = eff
    mean_eff_func = interp1d(mags,eff_df.mean(axis=1))
    std_eff_func =  interp1d(mags,eff_df.std(axis=1))

    return mean_eff_func,std_eff_func

def interpolate_zdf(zdf, marr):
    """Interpolate SFH data frame onto a log-linear mass grid, safely handling non-numeric columns."""
    # Ensure mass is numeric
    zdf['mass'] = pd.to_numeric(zdf['mass'], errors='coerce')

    # Split numeric and non-numeric columns
    numeric_cols = zdf.select_dtypes(include=[np.number]).columns
    non_numeric_cols = zdf.select_dtypes(exclude=[np.number]).columns

    # Group and take mean only for numeric columns
    gb = zdf.groupby(pd.cut(zdf['mass'], bins=marr))[numeric_cols].agg(np.nanmean)

    # Optionally, you could also keep the first value of non-numeric columns per bin
    for col in non_numeric_cols:
        gb[col] = zdf.groupby(pd.cut(zdf['mass'], bins=marr))[col].first()

    gb.dropna(subset=['mass'], inplace=True)
    gb.reset_index(drop=True, inplace=True)
    return gb

def make_z_pdf(zbins, power=2.5):
    """
    Create a normalized probability distribution for SN redshifts.

    Parameters
    ----------
    zbins : array-like
        Allowed redshift bin centers (must match flux_df keys).
    power : float
        Exponent for the distribution (z^power).

    Returns
    -------
    pdf : np.ndarray
        Normalized probabilities for each bin.
    """
    zbins = np.array(zbins, dtype=float)
    pdf = zbins**power
    pdf /= pdf.sum()
    return pdf