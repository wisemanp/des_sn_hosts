import numpy as np
from scipy.stats import norm


def age_rv_step(age, rv_young=3.0, rv_old=2.0, rv_sig_young=0.5, rv_sig_old=0.5, age_split=3):
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
    return (norm_young.rvs(size=len(age[age < age_split])) * (age < age_split)) + (
                norm_old.rvs(size=len(age[age > age_split])) * (age > age_split))


def mass_rv_step(mass, rv_low=3.0, rv_high=2.0, rv_sig_low=0.5, rv_sig_high=0.5, mass_split=10):
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
    return (norm_low.rvs(size=len(mass[mass < mass_split])) * (mass < mass_split)) + (
                norm_high.rvs(size=len(mass[mass > mass_split])) * (mass > mass_split))


def mass_rv_linear(mass, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, mass_fix_low=8, mass_fix_high=12):
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
    for rv_mu,rv_sig in zip(Rv_mus,Rv_sigs):
        Rvs.append(np.random.normal(Rv_mu,Rv_sig)
    return Rvs

def age_rv_linear(age, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, age_fix_low=0.1, age_fix_high=12):
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
    Rv_slope = (Rv_high-Rv_low) /(age_fix_high - age_fix_low)
    sig_slope = (Rv_sig_high-Rv_sig_low) /(age_fix_high - age_fix_low)
    Rv_mus = Rv_low +((age-age_fix_low) *Rv_slope)
    Rv_sigs = Rv_sig_low+((age - age_fix_low)*sig_slope)
    Rvs = []
    for rv_mu,rv_sig in zip(Rv_mus,Rv_sigs):
        Rvs.append(np.random.normal(Rv_mu,Rv_sig)
    return Rvs

def E_exp(tau,n):
    '''

    '''