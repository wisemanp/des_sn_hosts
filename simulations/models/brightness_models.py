from scipy.stats import norm
import numpy as np
from ..utils.HR_functions import add_age_step, add_mass_step


def tripp(alpha,beta,M0,sigma_int,mass_step,age_step,args):
    '''

    :param x1:
    :type x1:
    :param c:
    :type c:
    :param alpha:
    :type alpha:
    :param beta:
    :type beta:
    :param distmod:
    :type distmod:
    :param M0:
    :type M0:
    :param sigma_int:
    :type sigma_int:
    :param mass_step:
    :type mass_step:
    :param age_step:
    :type age_step:
    :param mass:
    :type mass:
    :param age:
    :type age:
    :return:
    :rtype:
    '''
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c'] - alpha*args['x1'] +\
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(np.log10(args['age']),age_step['mag'],age_step['loc'])

def tripp_rv(alpha,beta,M0,sigma_int,mass_step,age_step,args):
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc'])

def tripp_rv_popn_alpha_beta(mu_alpha,sig_alpha,mu_beta,sig_beta,M0,sigma_int,mass_step,age_step,args):
    alpha = norm(mu_alpha,sig_alpha).rvs(size=len(args['c']))
    beta = norm(mu_beta, sig_beta).rvs(size=len(args['c']))
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           +args['e_noise'] +add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc'])

def tripp_rv_two_beta_age(alpha,beta_young,beta_old,M0,sigma_int,mass_step,age_step,args):
    beta = (beta_old * (args['prog_age']=='old')) + (beta_young * (args['prog_age']=='young'))
    return M0 + args['distmod'] + norm(0, sigma_int).rvs(size=len(args['c'])) + beta * args['c_int'] - alpha * args[
        'x1'] + (args['rv'] + 1) * args['E'] + \
           add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc']) + add_age_step(
        args['SN_age'], age_step['mag'], age_step['loc'])

def tripp_rv_two_beta_popns_age(alpha,mu_beta_young,sig_beta_young,mu_beta_old,sig_beta_old,M0,sigma_int,mass_step,age_step,args):
    beta = np.array((norm(mu_beta_old,sig_beta_old).rvs(size=len(args['c'])) * (args['prog_age']=='old')) + (norm(mu_beta_young,sig_beta_young).rvs(size=len(args['c'])) * (args['prog_age']=='young')))

    return M0 + args['distmod'] + norm(0, sigma_int).rvs(size=len(args['c'])) + beta * np.array(args['c_int']) - alpha * np.array(args[
        'x1']) + (args['rv'] + 1) * args['E'] + \
           add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc']) + add_age_step(
        args['SN_age'], age_step['mag'], age_step['loc'])
