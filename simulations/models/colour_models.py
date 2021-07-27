from scipy.stats import norm
from .distributions import asymmetric_gaussian



def c_int_gauss(mu,sig,n=1):
    '''

    :param mu:
    :type mu:
    :param sig:
    :type sig:
    :return:
    :rtype:
    '''
    return norm(mu,sig).rvs(size=n)

def c_int_asymm(mu,sig_minus,sig_plus,n=1):
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
    cs = []
    for i in range(n):
        cs.append(asymmetric_gaussian(mu,sig_minus,sig_plus))
    return np.array(cs)

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
        c_ints = c_int_gauss(c_int_params['mu'],c_int_params['sig'],n=len(args['E']))
    elif c_int_type=='asymm':
        c_ints = c_int_asymm(c_int_params['mu'],c_int_params['sig_minus'],c_int_params['sig_plus'],n=len(args['E']))

    args['c'] = args['E']+c_ints
    if 'e_noise' in c_int_params.keys():
        args['e_noise'] =norm(0,c_int_params['e_noise']).rvs(size=len(args['E']))
        args['c'] = args['c'] + args['e_noise']
    else:
        args['e_noise'] = 0
    args['c_int'] = c_ints
    return args
