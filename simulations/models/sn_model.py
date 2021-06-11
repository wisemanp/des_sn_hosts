from .dust_models import age_rv_step, mass_rv_step, age_rv_linear, mass_rv_linear, E_exp, E_exp_mass, E_exp_age, random_rv, E_calc, E_from_host_random, E_two_component
from .colour_models import c_int_asymm, c_int_gauss, c_int_plus_dust
from .stretch_models import x1_int_asymm, x1_twogauss_age, x1_twogauss_fix
from .host_dust import choose_Av_SN_E_rv_fix, choose_Av_custom, choose_Av_SN_E_Rv_norm, choose_Av_SN_E_Rv_step
from .brightness_models import tripp, tripp_rv, tripp_rv_two_beta_age, tripp_rv_two_beta_popns_age, tripp_rv_popn_alpha_beta
class SN_Model():
    def __init__(self):
        '''

        '''
        pass


    def age_rv_step(self,args,params):
        return age_rv_step(args['age'], params['rv_young'], params['rv_old'], params['rv_sig_young'], params['rv_sig_old'], params['age_split'])

    def mass_rv_step(self,args,params):
        return mass_rv_step(args['mass'], params['rv_low'], params['rv_high'], params['rv_sig_low'], params['rv_sig_high'], params['mass_split'])

    def mass_rv_linear(self,args,params ):
        return mass_rv_linear(args['mass'], params['rv_low'], params['rv_high'], params['rv_sig_low'], params['rv_sig_high'], params['mass_fix_low'], params['mass_fix_high'])

    def age_rv_linear(self,args,params):
        return age_rv_linear(args['age'], params['rv_low'], params['rv_high'], params['rv_sig_low'], params['rv_sig_high'], params['age_fix_low'], params['age_fix_high'])

    def E_exp(self,args,params):
        return E_exp(params['TauE'],args['n'])

    def E_exp_mass(self,args,params):
        return E_exp_mass(args['mass'],params['Tau_low'],params['Tau_high'],params['mass_split'])

    def E_exp_age(self,args,params):
        return E_exp_age(args['age'],params['Tau_low'],params['Tau_high'],params['age_split'])

    def random_rv(self,args,params):
        return random_rv(params['Rv_mu'],params['Rv_sig'],args['n'])

    def E_calc(self,args,params):
        return E_calc(args['host_Av'],params['Rv'])

    def E_from_host_random(self,args,params):
        return E_from_host_random(args['host_Av'],params['Av_sig'],args['Rv'],params['Rv_sig'])

    def E_two_component(self,args,params):
        return E_two_component(params['TauE_int'],params['Av_host'],params['Rv_host'],params['Av_sig_host'],params['Rv_sig_host'],n=args['n'])

    def x1_int_asymm(self,args,params):
        args['x1'] = x1_int_asymm(['mu'],['sig_minus'],['sig_plus'],args['n'])
        return args

    def x1_twogauss_fix(self,args,params):
        args['x1'] = x1_twogauss_fix(params['mu_low'],params['sig_low'],params['mu_high'],params['sig_high'],params['frac_low'],args['n'])
        return args

    def x1_twogauss_age(self,args,params):
        sampler = x1_twogauss_age(params['mu_old'],params['sig_old'],params['mu_young'],params['sig_young'],params['age_step_loc'],params['old_prob'])
        args['x1'],args['prog_age'] =  sampler.sample(args['SN_age'])
        return args
    def choose_Av_SN_E_rv_fix(self,args,params):
        return choose_Av_SN_E_rv_fix(args['Av_grid'],args['E'],params['Rv'],params['Av_sig'])

    def choose_Av_SN_E_Rv_norm(self,args,params):
        return choose_Av_SN_E_Rv_norm(args['Av_grid'],args['E'],params['Rv_mu'],params['Rv_sig'],params['Av_sig'])

    def choose_Av_SN_E_Rv_step(self,args,params):
        return choose_Av_SN_E_Rv_step(args['Av_grid'],args['E'],args['mass'],params['Rv_mu_low'],params['Rv_mu_high'],params['Rv_sig_low'],params['Rv_sig_high'],params['Av_sig'])

    def choose_Av_custom(self,args,params):
        return choose_Av_custom(args['Av_grid'],params['dist'],args['n'])

    def c_int_gauss(self,args,params):
        return c_int_gauss(params['mu'],params['sig'],args['n'])

    def c_int_asymm(self,args,params):
        return c_int_asymm(params['mu'],params['sig_minus'],params['sig_plus'],args['n'])

    def c_int_plus_dust(self,args,params):
        return c_int_plus_dust(args,params['c_int_type'],params['c_int_params'])

    def tripp(self,args,params):
        return tripp(params['alpha'],params['beta'],params['M0'],params['sigma_int'],params['mass_step'],params['age_step'],args)

    def tripp_rv(self,args,params):
        return tripp_rv(params['alpha'],params['beta'],params['M0'],params['sigma_int'],params['mass_step'],params['age_step'],args)

    def tripp_rv_popn_alpha_beta(self,args,params):
        return tripp_rv_popn_alpha_beta(params['mu_alpha'],params['sig_alpha'],params['mu_beta'],params['sig_beta'],params['M0'],params['sigma_int'],params['mass_step'],params['age_step'],args)

    def tripp_rv_two_beta_age(self,args,params):
        return tripp_rv_two_beta_age(params['alpha'],params['beta_young'],params['beta_old'],params['M0'],params['sigma_int'],params['mass_step'],params['age_step'],args)

    def tripp_rv_two_beta_popns_age(self,args,params):
        return tripp_rv_two_beta_popns_age(params['alpha'], params['beta_young'],params['sig_beta_young'], params['beta_old'], params['sig_beta_old'], params['M0'],
                                     params['sigma_int'], params['mass_step'], params['age_step'], args)


