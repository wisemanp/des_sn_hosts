import numpy as np
from scipy.stats import norm
def choose_Av_SN_E_rv_fix(Av_grid,Es,Rv,Av_sig):
    '''

    :param Av_sig:
    :type Av_sig:
    :param Av_grid:
    :type Av_grid:
    :param Es:
    :type Es:
    :param Rv:
    :type Rv:
    :return: avs
    :rtype:
    '''
    p_array = norm(Es/Rv,Av_sig).pdf(np.array(len(Es)*[Av_grid,]).T)
    avs = [np.random.choice(Av_grid,p=p_array[:,i]/np.sum(p_array[:,i])) for i in range(p_array.shape[1])]
    return avs

def choose_Av_SN_E_Rv_norm(Av_grid,Es,Rv_mu,Rv_sig,Av_sig):
    '''

    :param Rv_mu:
    :type Rv_mu:
    :param Rv_sig:
    :type Rv_sig:
    :param Av_sig:
    :type Av_sig:
    :param Av_grid:
    :type Av_grid:
    :param Es:
    :type Es:
    :return: avs
    :rtype:
    '''
    Rv = norm(Rv_mu,Rv_sig).rvs(len(Es))
    p_array = norm(Es/Rv,Av_sig).pdf(np.array(len(Es)*[Av_grid,]).T)
    p_array = np.clip(p_array,a_min=0.001,a_max=None)
    avs = np.array([np.random.choice(Av_grid,p=p_array[:,i]/np.sum(p_array[:,i])) for i in range(p_array.shape[1])])
    return avs

def choose_Av_SN_E_Rv_step(Av_grid,Es,mass,Rv_mu_low,Rv_mu_high,Rv_sig_low,Rv_sig_high,Av_sig):
    '''

    :param Rv_mu:
    :type Rv_mu:
    :param Rv_sig:
    :type Rv_sig:
    :param Av_sig:
    :type Av_sig:
    :param Av_grid:
    :type Av_grid:
    :param Es:
    :type Es:
    :return: avs
    :rtype:
    '''

    norm_low = norm(Rv_mu_low, Rv_sig_low)
    norm_high = norm(Rv_mu_high, Rv_sig_high)
    Rv = (norm_low.rvs(size=len(mass)) * (mass < mass_split)) + (
            norm_high.rvs(size=len(mass)) * (mass > mass_split))
    p_array = norm(Es/Rv,Av_sig).pdf(np.array(len(Es)*[Av_grid,]).T)
    avs = np.array([np.random.choice(Av_grid,p=p_array[:,i]/np.sum(p_array[:,i])) for i in range(p_array.shape[1])])
    return avs

def choose_Av_custom(Av_grid,dist,n=1):
    p_array = dist.pdf(np.array(n * [Av_grid, ]).T)
    avs = np.array([np.random.choice(Av_grid, p=p_array[:, i] / np.sum(p_array[:, i])) for i in range(p_array.shape[1])])
    return avs
