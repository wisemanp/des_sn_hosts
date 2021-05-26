import numpy as np
from scipy.stats import norm

def age_rv_step(age,rv_young=3.0,rv_old=2.0,rv_sig_young=0.5,rv_sig_old=0.5,age_split=3):
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
    norm_young = norm(rv_young,rv_sig_young)
    norm_old = norm(rv_old,rv_sig_old)
    return (norm_young.rvs(size=len(age[age<age_split])) * (age<age_split)) + (norm_old.rvs(size=len(age[age>age_split])) * (age>age_split))

def mass_rv_step(mass,rv_low=3.0,rv_high=2.0,rv_sig_low=0.5,rv_sig_high=0.5,mass_split=10):
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
    norm_low = norm(rv_low,rv_sig_low)
    norm_high = norm(rv_high,rv_sig_high)
    return (norm_low.rvs(size=len(mass[mass<mass_split])) * (mass<mass_split)) + (norm_high.rvs(size=len(mass[mass>mass_split])) * (mass>mass_split))