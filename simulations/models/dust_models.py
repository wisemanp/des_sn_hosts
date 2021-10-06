import numpy as np
from scipy.stats import norm, expon

law_Rvs ={
    'C00':4.05
}

def age_rv_step(age, rv_young=3.0, rv_old=2.0, rv_sig_young=0.5, rv_sig_old=0.5, age_split=3,rv_min=1.2):
    """

    :param age:
    :param rv_sig_young:
    :param rv_sig_old:
    :param age_split:
    :param rv_young:
    :param rv_old:
    :return: Rvs
    :rtype array
    """
    norm_young = norm(rv_young, rv_sig_young)
    norm_old = norm(rv_old, rv_sig_old)
    return np.clip((norm_young.rvs(size=len(age)) * (age < age_split)) + (
                norm_old.rvs(size=len(age)) * (age > age_split)),a_min=rv_min,a_max=None)


def mass_rv_step(mass, rv_low=3.0, rv_high=2.0, rv_sig_low=0.5, rv_sig_high=0.5, mass_split=10,rv_min=1.2):
    """

    :param age:
    :param rv_sig_young:
    :param rv_sig_old:
    :param age_split:
    :param rv_young:
    :param rv_old:
    :return: Rvs
    :rtype array
    """
    norm_low = norm(rv_low, rv_sig_low)
    norm_high = norm(rv_high, rv_sig_high)
    return np.clip((norm_low.rvs(size=len(mass)) * (mass < mass_split)) + (
                norm_high.rvs(size=len(mass)) * (mass > mass_split)),a_min=rv_min,a_max=None)


def mass_rv_linear(mass, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, mass_fix_low=8, mass_fix_high=12,rv_min=1.2):
    '''
    :param mass:
    :param Rv_low:
    :param Rv_high:
    :param rv_sig_low:
    :param rv_sig_high:
    :param mass_fix_low:
    :param mass_fix_high:
    :return: Rvs
    :rtype array
    '''
    Rv_slope = (Rv_high-Rv_low) /(mass_fix_high - mass_fix_low)
    sig_slope = (Rv_sig_high-Rv_sig_low) /(mass_fix_high - mass_fix_low)
    Rv_mus = Rv_low +((mass-mass_fix_low) *Rv_slope)
    Rv_sigs = Rv_sig_low+((mass - mass_fix_low)*sig_slope)
    Rvs = []
    for Rv_mu,Rv_sig in zip(Rv_mus,Rv_sigs):
        Rvs.append(np.random.normal(Rv_mu,Rv_sig))
    return np.clip(np.array(Rvs),a_min=rv_min,a_max=None)

def age_rv_linear(age, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, age_fix_low=0.1, age_fix_high=12,rv_min=1.2):
    '''
    :param age:
    :param Rv_low:
    :param Rv_high:
    :param Rv_sig_low:
    :param Rv_sig_high:
    :param age_fix_low:
    :param age_fix_high:
    :return: Rvs
    :rtype array
    '''
    logage = np.log10(age)
    log_age_fix_low = np.log10(age_fix_low)
    log_age_fix_high = np.log10(age_fix_high)
    Rv_slope = (Rv_high-Rv_low) /(age_fix_high - age_fix_low)
    sig_slope = (Rv_sig_high-Rv_sig_low) /(age_fix_high - age_fix_low)
    Rv_mus = Rv_low +((logage-age_fix_low) *Rv_slope)
    Rv_sigs = Rv_sig_low+((logage - age_fix_low)*sig_slope)
    Rvs = []
    for rv_mu,rv_sig in zip(Rv_mus,Rv_sigs):
        Rvs.append(np.random.normal(rv_mu,rv_sig))
    return np.clip(np.array(Rvs),a_min=rv_min,a_max=None)

def E_exp(TauE,n=1):
    '''
    Returns n values of the reddening E(B-V) for a given mean reddening tau.
    :param tau:
    :type tau: float
    :param n:
    :type n: int
    :return: E
    :rtype: array
    '''
    E = expon(scale=TauE,size=n)
    return E

def E_exp_mass(mass,Tau_low,Tau_high,mass_split=10):
    '''

    :param mass:
    :type mass:
    :param Tau_low:
    :type Tau_low:
    :param Tau_high:
    :type Tau_high:
    :return:
    :rtype:
    '''
    mass = np.log10(mass)
    E_low = expon(scale=Tau_low)
    E_high = expon(scale=Tau_high)
    return (E_low.rvs(size=len(mass)) * (mass < mass_split)) + (
            E_high.rvs(size=len(mass)) * (mass > mass_split))

def E_exp_age(age,Tau_low,Tau_high,age_split=3):
    '''

    :param age:
    :type age:
    :param age_split:
    :type age_split:
    :param Tau_low:
    :type Tau_low:
    :param Tau_high:
    :type Tau_high:
    :return:
    :rtype:
    '''

    E_low = expon(scale=Tau_low)
    E_high = expon(scale=Tau_high)
    return (E_low.rvs(size=len(age)) * (age < age_split)) + (
            E_high.rvs(size=len(age)) * (age > age_split))

def random_rv(Rv_mu,Rv_sig,n):
    Rv_norm = norm(Rv_mu,Rv_sig)
    return Rv_norm.rvs(size=n)

def E_calc(Av,Rv):
    '''

    :param Av:
    :type Av: array-like
    :param Rv:
    :type Rv: array-like
    :return: E
    :rtype: array-like
    '''
    E = Av*Rv
    return E

def E_from_host_random(Av,Av_sig=0.25,Rv=4.05,Rv_sig=0.5):
    '''

    :param Rv_sig: standard deviation of Rv distribution to draw from
    :type Rv_sig: float or array of float
    :param Rv: dust law slope Rv
    :type Rv: array of float
    :param Av_sig: standard deviation of Av distirbution to draw from
    :type Av_sig: float or array of float
    :param Av:V-band extinction Av
    :type Av: array
    :return: E
    :rtype: array-like
    '''
    E = []
    for A,R in zip(Av,Rv):
        E.append(np.random.norm(A,Av_sig)/np.random.norm(R,Rv_sig))
    return E

def E_two_component(TauE_int, Av_host,Rv_host,Av_sig_host,Rv_sig_host,n=1):
    '''
    Two-component reddening
    :param TauE_int:
    :type TauE_int:
    :param Rv_int:
    :type Rv_int:
    :param Av_host:
    :type Av_host:
    :param Rv_host:
    :type Rv_host:
    :param Av_sig_host:
    :type Av_sig_host:
    :param Rv_sig_host:
    :type Rv_sig_host:
    :return:
    :rtype:
    '''
    E_int = E_exp(TauE_int,n)
    E_host = E_from_host_random(Av_host,Av_sig_host,Rv_host,Rv_sig_host)
    return E_int + E_host
