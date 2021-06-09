'''Galaxy-related functions'''
import numpy as np
def double_schechter(logM,logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):#logM_star,phi_star_1,alpha_1,phi_star_2,alpha_2):
    if phi_star_1<0:
        phi_star_1 = 10**phi_star_1
    if phi_star_2<0:
        phi_star_2 = 10**phi_star_2
    return np.log(10)*((phi_star_1*(10**((logM - logM_star)*(1+alpha_1))))+(phi_star_2*(10**((logM - logM_star)*(1+alpha_2)))))*\
                np.exp(-10**(logM-logM_star))

def single_schechter(logM,logM_star,phi_star,alpha,):
    return np.log(10)*((phi_star*(10**((logM - logM_star)*(1+alpha))))*\
                np.exp(-10**(logM-logM_star)))

zfourge = {
    0.2: {'logM_star':10.78,'phi_star_1':-2.54,'alpha_1':-0.98,'phi_star_2':-4.29,'alpha_2':-1.9},
    0.5: {'logM_star':10.7,'phi_star_1':-2.55,'alpha_1':-0.39,'phi_star_2':-3.15,'alpha_2':-1.53},
    0.75: {'logM_star':10.66,'phi_star_1':-2.56,'alpha_1':-0.37,'phi_star_2':-3.49,'alpha_2':-1.62},
    1: {'logM_star':10.54,'phi_star_1':-2.72,'alpha_1':0.3,'phi_star_2':-3.17,'alpha_2':-1.45},
}

def schechter(z,logM):
    if z<0.5:

        return double_schechter(logM,**zfourge[0.2])
    elif z>=0.5 and z<0.75:
        return double_schechter(logM,**zfourge[0.5])
    elif z>0.75 and z<1:
        return double_schechter(logM,**zfourge[0.75])
    elif z>1 and z<1.25:
        return double_schechter(logM,**zfourge[1])
