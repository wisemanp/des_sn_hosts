from dust_models import age_rv_step, mass_rv_step, age_rv_linear, mass_rv_linear, E_exp, E_exp_mass, E_exp_age, Rv_random, E_calc, E_from_host_random, E_two_component
from colour_models import c_int_asymm, c_int_gauss, c_int_plus_dust
from stretch_models import x1_int_asymm, x1_twogauss_age
from host_dust import choose_Av_SN_E_rv_fix, choose_Av_custom, choose_Av_SN_E_Rv_norm, choose_Av_SN_E_Rv_step

class SN_Model():
    def __init__(self,config):
        '''

        '''
        self.config = config

    def age_rv_step(self,age, rv_young=3.0, rv_old=2.0, rv_sig_young=0.5, rv_sig_old=0.5, age_split=3):
        return age_rv_step(age, rv_young, rv_old, rv_sig_young, rv_sig_old, age_split)

    def mass_rv_step(self,mass, rv_low=3.0, rv_high=2.0, rv_sig_low=0.5, rv_sig_high=0.5, mass_split=10):
        return mass_rv_step(mass, rv_low, rv_high, rv_sig_low, rv_sig_high, mass_split)

    def mass_rv_linear(self, mass, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, mass_fix_low=8, mass_fix_high=12):
        return mass_rv_linear(mass, Rv_low, Rv_high, Rv_sig_low, Rv_sig_high, mass_fix_low, mass_fix_high)

    def age_rv_linear(self,age, Rv_low=3.0, Rv_high=2.0, Rv_sig_low=0.5, Rv_sig_high=1, age_fix_low=0.1, age_fix_high=12):
        return age_rv_linear(age, Rv_low, Rv_high, Rv_sig_low, Rv_sig_high, age_fix_low, age_fix_high)

    def E_exp(self,TauE,n=1):
        return E_exp(TauE,n)

    def E_exp_mass(self,Tau_low,Tau_high,mass_split=10):
        return E_exp_mass(mass,Tau_low,Tau_high,mass_split)

    def E_exp_age(self,age,Tau_low,Tau_high,age_split=3):
        return E_exp_age(age,Tau_low,Tau_high,age_split)

    def Rv_random(self,Rv_mu,Rv_sig,n):
        return Rv_random(Rv_mu,Rv_sig,n)

    def E_calc(self,Av,Rv):
        return E_calc(Av,Rv)
    def E_from_host_random(self,Av,Av_sig=0.25,Rv=4.05,Rv_sig=0.5):
        return E_from_host_random(Av,Av_sig,Rv,Rv_sig)

    def E_two_component(self,TauE_int, Rv_int,Av_host,Rv_host,Av_sig_host,Rv_sig_host,n=1):
        return E_two_component(TauE_int, Rv_int,Av_host,Rv_host,Av_sig_host,Rv_sig_host,n)