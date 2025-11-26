from scipy.stats import norm
import numpy as np
from ..utils.HR_functions import add_age_step, add_mass_step, add_galage_grad, add_SN_age_grad


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
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*np.array(args['c']) - alpha*np.array(args['x1']) +\
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(np.log10(args['SN_age']),age_step['mag'],age_step['loc']), alpha, beta

def tripp_no_scatter(alpha,beta,M0,args):

    return M0 + args['distmod'] + beta*np.array(args['c']) - alpha*np.array(args['x1']), alpha, beta


def tripp_rv(alpha,beta,M0,sigma_int,mass_step,age_step,args):
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc']),    alpha, beta

def tripp_rv_popn_alpha_beta(mu_alpha,sig_alpha,mu_beta,sig_beta,M0,sigma_int,mass_step,age_step,args):
    alpha = norm(mu_alpha,sig_alpha).rvs(size=len(args['c']))
    beta = norm(mu_beta, sig_beta).rvs(size=len(args['c']))
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc']),   alpha,  beta

def tripp_rv_agebias_fix_alpha_beta(mu_alpha,mu_beta,M0,sigma_int,mass_step,age_step,galage_grad,SNage_grad,args):
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + mu_beta*args['c_int'] - mu_alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc']) + \
            add_galage_grad(args['mean_ages']/1000,galage_grad['slope'],galage_grad['intercept']) + \
                add_SN_age_grad(args['SN_age'],SNage_grad['slope'],SNage_grad['intercept']),  mu_alpha,mu_beta

def tripp_rv_agebias_popn_alpha_beta(mu_alpha,sig_alpha,mu_beta,sig_beta,M0,sigma_int,mass_step,age_step,galage_grad,SNage_grad,args):
    alpha = norm(mu_alpha,sig_alpha).rvs(size=len(args['c']))
    beta = norm(mu_beta, sig_beta).rvs(size=len(args['c']))
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc']) + \
            add_galage_grad(args['mean_ages']/1000,galage_grad['slope'],galage_grad['intercept']) + \
                add_SN_age_grad(args['SN_age'],SNage_grad['slope'],SNage_grad['intercept']),   alpha,  beta


def tripp_rv_age_alpha_popn_beta(mu_alpha_young,sig_alpha_young,mu_alpha_old,sig_alpha_old,
                                        mu_beta,sig_beta,M0,sigma_int,mass_step,age_step,args):
    args['prog_age'] = np.array(args['prog_age'])
    alpha = np.array((norm(mu_alpha_old,sig_alpha_old).rvs(size=len(args['c'])) * (args['prog_age']=='old')) + (norm(mu_alpha_young,sig_alpha_young).rvs(size=len(args['c'])) * (args['prog_age']=='young')))
    beta = norm(mu_beta, sig_beta).rvs(size=len(args['c']))
    #FIXME!!!
    return M0 + args['distmod'] + norm(0,sigma_int).rvs(size=len(args['c'])) + beta*args['c_int'] - alpha*args['x1'] + (args['rv']+1)*args['E'] + \
           add_mass_step(np.log10(args['mass']),mass_step['mag'],mass_step['loc']) + add_age_step(args['SN_age'],age_step['mag'],age_step['loc']),    alpha, beta


def tripp_rv_two_beta_age(alpha,beta_young,beta_old,M0,sigma_int,mass_step,age_step,args):
    beta = (beta_old * (args['prog_age']=='old')) + (beta_young * (args['prog_age']=='young'))
    return M0 + args['distmod'] + norm(0, sigma_int).rvs(size=len(args['c'])) + beta * args['c_int'] - alpha * args[
        'x1'] + (args['rv'] + 1) * args['E'] + \
           add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc']) + add_age_step(
        args['SN_age'], age_step['mag'], age_step['loc']), alpha, beta

def tripp_rv_two_beta_popns_age(alpha,mu_beta_young,sig_beta_young,mu_beta_old,sig_beta_old,M0,sigma_int,mass_step,age_step,args):
    beta = np.array((norm(mu_beta_old,sig_beta_old).rvs(size=len(args['c'])) * (args['prog_age']=='old')) + (norm(mu_beta_young,sig_beta_young).rvs(size=len(args['c'])) * (args['prog_age']=='young')))

    return M0 + args['distmod'] + norm(0, sigma_int).rvs(size=len(args['c'])) + beta * np.array(args['c_int']) - alpha * np.array(args[
        'x1']) + (args['rv'] + 1) * args['E'] + \
           add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc']) + add_age_step(
        args['SN_age'], age_step['mag'], age_step['loc']), alpha, beta

def tripp_rv_two_beta_popns_age2(alpha,mu_beta_young,sig_beta_young,mu_beta_old,sig_beta_old,M0,sigma_int,mass_step,age_step,args):
    beta = np.array((norm(mu_beta_old,sig_beta_old).rvs(size=len(args['c'])) * (args['SN_age']>age_step['loc'])) + \
        (norm(mu_beta_young,sig_beta_young).rvs(size=len(args['c'])) * (args['SN_age']<=age_step['loc'])))

    return M0 + args['distmod'] + norm(0, sigma_int).rvs(size=len(args['c'])) + beta * np.array(args['c_int']) - alpha * np.array(args[
        'x1']) + (args['rv'] + 1) * args['E'] + \
           add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc']) + add_age_step(
        args['SN_age'], age_step['mag'], age_step['loc']), alpha, beta

def tripp_rv_popn_alpha_beta_z_lin(mu_alpha, sig_alpha, mu_beta, sig_beta,
                                   M0, sigma_int, mass_step, age_step,
                                   gamma_z, z_ref, args):
    """
    Tripp-like model with redshift-evolving absolute magnitude:
      M0(z) = M0 + gamma_z * (z - z_ref)

    Params:
      mu_alpha, sig_alpha, mu_beta, sig_beta, M0, sigma_int, mass_step, age_step
      gamma_z: slope of M0 evolution per unit z
      z_ref: reference redshift for zero evolution offset
    """
    n = len(args['c'])
    z_arr = np.asarray(args.get('z', np.zeros(n)), dtype=float)
    M0_z = M0 + gamma_z * (z_arr - z_ref)

    alpha = norm(mu_alpha, sig_alpha).rvs(size=n)
    beta = norm(mu_beta, sig_beta).rvs(size=n)

    mB = (
        M0_z
        + args['distmod']
        + norm(0, sigma_int).rvs(size=n)
        + beta * np.array(args['c_int'])
        - alpha * np.array(args['x1'])
        + (np.array(args['rv']) + 1) * np.array(args['E'])
        + add_mass_step(np.log10(args['mass']), mass_step['mag'], mass_step['loc'])
        + add_age_step(args['SN_age'], age_step['mag'], age_step['loc'])
    )
    return mB, alpha, beta
