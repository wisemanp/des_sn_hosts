import numpy as np

def P_Edust(E,TauE):
    '''Exponential distribution of dust reddening'''
    return (1/TauE)*np.exp(-1*E/TauE)

def asymmetric_gaussian(x,mu,sig_minus,sig_plus):
    '''An asymmetric gaussian defined by mean mu, lower std sig_minus and upper std sig_plus'''
    return ((x>mu)*np.exp(-(x-mu)**2/(2*sig_plus**2))) + ((x<mu)*np.exp(-(x-mu)**2/(2*sig_minus**2)))
