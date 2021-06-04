import numpy as np
from scipy.stats import norm
from distributions import asymmetric_gaussian
def x1_int_asymm(mu,sig_minus,sig_plus,n=1):
    '''
    Sample x1 from an asymmetric Gaussian of mean mu, lower std sig_minus, upper std sig_plus
    :param mu:
    :type mu:
    :param sig_minus:
    :type sig_minus:
    :param sig_plus:
    :type sig_plus:
    :return: x1s
    :rtype: array
    '''
    x1s = []
    for i in range(n):
        x1s.append(asymmetric_gaussian(x1_grid,mu,sig_minus,sig_plus))
    return np.array(x1s)

def x1_twogauss_fix(mu_low,sig_low,mu_high,sig_high,frac_low=0.5,n=1):
    '''

    '''
    norm_low = norm(mu_low,sig_low)
    norm_high = norm(mu_high,sig_high)
    fracs = [frac_low,1-frac_low]
    x1s []
    for i in range(n):
        x1s.append(np.random.choice([norm_low.rvs(),norm_high.rvs()],p=fracs))
    return np.array(x1s)

class x1_twogauss_age():
    '''

    '''
    def __init__(self,mu_old,sig_old,mu_young,sig_young,age_step,old_prob=0.5):
        self._set_norm_old(mu_old,sig_old)
        self._set_norm_young(mu_young,sig_young)
        self.age_step = age_step
        self.old_prob = old_prob
    def _set_norm_old(self,mu_old,sig_old):
        self.norm_old = norm(mu_old,sig_old)
    def _set_norm_young(self,mu_young,sig_young):
        self.norm_young = norm(mu_young,sig_young)

    def sample(self,ages,old_probs=[],return_prog_age=True):
        if len(old_probs)==0:
            old_probs = [self.old_prob,1-self.old_prob]
        x1s = []
        prog_age_choices = []
        for counter,age in enumerate(ages):
            if age > self.age_step:
                x1_rand = {'old':self.norm_old.rvs(),'young':self.norm_young.rvs()}
                prog_age_choice = np.random.choice(['old','young'],p=old_probs)
                x1s.append(x1_rand[prog_age_choice])
            elif age < self.age_step:
                prog_age_choice = 'young'
                x1.append(self.norm_young.rvs())
            prog_age_choices.append(prog_age_choice)
        if return_prog_age:
            return x1s,prog_age_choices
        else:
            return x1s

