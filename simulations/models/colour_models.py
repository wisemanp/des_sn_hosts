from scipy.stats import norm
from .distributions import asymmetric_gaussian
import numpy as np


def c_int_gauss(args,mu,sig,):
    '''

    :param mu:
    :type mu:
    :param sig:
    :type sig:
    :return:
    :rtype:
    '''
    n = args['n']
    args['c'] = norm(mu,sig).rvs(size=n)
    return args

def c_int_asymm(args, mu,sig_minus,sig_plus,):
    '''

    :param mu:
    :type mu:
    :param sig_minus:
    :type sig_minus:
    :param sig_plus:
    :type sig_plus:
    :return:
    :rtype:
    '''
    n = args['n']
    cs = []
    for i in range(n):
        cs.append(asymmetric_gaussian(mu,sig_minus,sig_plus))
    args['c'] = np.array(cs)
    return args

def c_int_plus_dust(args,c_int_type,c_int_params):
    '''

    :param n:
    :type n:
    :param Es:
    :type Es:
    :return:
    :rtype:
    '''
    if c_int_type=='norm':
        args = c_int_gauss(args, c_int_params['mu'],c_int_params['sig'],)
    elif c_int_type=='asymm':
        args = c_int_asymm(args, c_int_params['mu'],c_int_params['sig_minus'],c_int_params['sig_plus'],)
    args['c_int'] = args['c'].copy()
    args['c'] = args['E']+args['c_ints']
    
    return args
