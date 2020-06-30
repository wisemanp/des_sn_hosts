
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u

def compute_HC(DLR,SEP):
    dlr = np.sort(DLR) # first sort DLR array
    e = 1e-5 # define epsilon, some small number
    if len(dlr)==1: # only 1 object in radius
        HC = -99.0
    else:
        delta12 = dlr[1] - dlr[0] + e
        D1D2 = dlr[0]**2/dlr[1] + e
        Dsum = 0
        for i in range(0, len(dlr)):
            for j in range(i+1, len(dlr)):
                didj = dlr[i]/dlr[j] + e
                delta_d = dlr[j] - dlr[i] + e
                Dsum += didj/((i+1)**2*delta_d)
        HC = np.log10(D1D2*Dsum/delta12)
    return HC

def compute_features(DLR,SEP):
    dlr = np.sort(DLR) # first sort DLR array
    sep = np.sort(SEP)
    e = 1e-5 # define epsilon, some small number
    if len(dlr)==1: # only 1 object in radius
        HC = -99.0
        D12 = D1D2 = D13 = D1D3 = S12 = S12_DLR = S1_S2 = S13 = S1_S3 = 0
    else:
        # closest match
        delta12 = dlr[1] - dlr[0] + e
        D12 =dlr[1] - dlr[0]
        D1D2 = dlr[0]**2/dlr[1] + e
        D1_D2 = dlr[0]/dlr[1]
         # sep
        DLR_sorted_SEP = sep[np.argsort(DLR)]
        S12_DLR = DLR_sorted_SEP[1] - DLR_sorted_SEP[0]
        S12 = sep[1] - sep[0]
        S1_S2 = sep[0]/sep[1]

        D13 = dlr[2] - dlr[0]
        D1_D3 = dlr[0]/dlr[2]
        S13 = sep[2] - sep[0]
        S1_S3 = sep[0]/sep[2]
        Dsum = 0
        for i in range(0, len(dlr)):
            for j in range(i+1, len(dlr)):
                didj = dlr[i]/dlr[j] + e
                delta_d = dlr[j] - dlr[i] + e
                Dsum += didj/((i+1)**2*delta_d)

        HC = np.log10(D1D2*Dsum/delta12)
    return HC, D12, D1_D2, D13, D1_D3, S12, S12_DLR, S1_S2, S13, S1_S3

def place_sn(galaxy,d_DLR):
    '''Function for putting an SN at a certain location from a host, given a DLR '''
    rad  = np.pi/180                   # convert deg to rad
    pix_arcsec = 0.264                 # pixel scale (arcsec per pixel)
    pix2_arcsec2 = 0.264**2            # pix^2 to arcsec^2 conversion factor
    pix2_deg2 = pix2_arcsec2/(3600**2) # pix^2 to deg^2 conversion factor
    RA_GAL, DEC_GAL, A_IMAGE, B_IMAGE, THETA_IMAGE = galaxy[['RA','DEC','A_IMAGE','B_IMAGE','THETA_IMAGE']]
    A_ARCSEC = A_IMAGE*pix_arcsec
    B_ARCSEC = B_IMAGE*pix_arcsec
    GAMMA = np.random.uniform(0,2*np.pi) # angle between RA-axis and SN-host vector

    PHI = np.radians(THETA_IMAGE) - GAMMA # angle between semi-major axis of host and SN-host vector

    rPHI = A_ARCSEC*B_ARCSEC/np.sqrt((A_ARCSEC*np.sin(PHI))**2 +
                                     (B_ARCSEC*np.cos(PHI))**2)
    angsep = d_DLR *rPHI
    y = (1/3600)*angsep*np.cos(GAMMA)
    DEC_SN = DEC_GAL + y
    x = y*np.tan(GAMMA)
    RA_SN = RA_GAL + (x/np.cos(DEC_SN*rad))
    return RA_SN,DEC_SN

def get_edge_flags(xs,ys,dist=20):
    '''Flags objects that are near the edge of a chip'''

    flags = np.zeros_like(xs)
    for counter,x in enumerate(xs):
        if x<20 or x>4080:
            flags[counter]=1
    for counter,y in enumerate(ys):
        if y<20 or y>2080:
            flags[counter]=1
    return flags
class Constants():
    def __init__(self):
        self.Lsun = 3.838E+33
        self.fluxlim_Jy_des = 1E-7
        self.des_filters = {
            'g': [3920, 4775.0, 5630],
            'r': [5330, 6215.0, 7100],
            'i': [6710, 7540.0, 8370],
            'z': [7930, 8690.0, 9450]
            }
        self.fluxlim_ergcms_des = (self.des_filters['i'][2]-self.des_filters['i'][0])*2.99792458E-05*self.fluxlim_Jy_des/(self.des_filters['i'][1]**2)
        self.des_area_frac = 27/41252.96125
        self.V_co_1 = cosmo.comoving_volume(1)
        self.D08_Mpc = lam = (2.6 * 1E-5)/(u.Mpc**3)
        self.D08_z = lam*self.V_co_1
