''' Spectroscopic utils courtesy of Mike Childress '''

import scipy.interpolate
import scipy.optimize
from pysynphot import observation
from pysynphot import spectrum
import numpy as np
import pandas as pd 
#------------------------------------------------------------------------
# spectrum and filter class definitions
class Spectrum(object):
    #---------------------------------------------
    def __init__(self, wave, flux, var=None):
        # initialize values
        self.__wave = wave
        self.__flux = flux
        if type(var) == type(None):
            self.__var = np.zeros(len(wave), dtype='d')
        else:
            self.__var = var
        # set up scipy interpolate object
        self.__flux_interp = scipy.interpolate.interp1d(
            self.__wave, self.__flux, bounds_error=False, fill_value=0.0)
        self.__var_interp = scipy.interpolate.interp1d(
            self.__wave, self.__var, bounds_error=False, fill_value=0.0)

    #---------------------------------------------
    def flux(self, wave_array = None,
             wmin=None, wmax=None):
        if type(wave_array) == type(None):
            init_wave = self.__wave
            if wmin == None:
                wmin = init_wave[0]
            if wmax == None:
                wmax = init_wave[-1]
            i_wmin = np.nonzero(init_wave >= wmin)[0][0]
            i_wmax = np.nonzero(init_wave <= wmax)[0][-1]
            return self.__flux[i_wmin:i_wmax]
        else:
            return self.__flux_interp(wave_array)

    def var(self, wave_array = None,
            wmin=None, wmax=None):
        if type(wave_array) == type(None):
            init_wave = self.__wave
            if wmin == None:
                wmin = init_wave[0]
            if wmax == None:
                wmax = init_wave[-1]
            i_wmin = np.nonzero(init_wave >= wmin)[0][0]
            i_wmax = np.nonzero(init_wave <= wmax)[0][-1]
            return self.__var[i_wmin:i_wmax]
        else:
            return self.__var_interp(wave_array)

    def wave(self, wmin=None, wmax=None):
        init_wave = self.__wave
        if wmin == None:
            wmin = init_wave[0]
        if wmax == None:
            wmax = init_wave[-1]
        i_wmin = np.nonzero(init_wave >= wmin)[0][0]
        i_wmax = np.nonzero(init_wave <= wmax)[0][-1]
        return init_wave[i_wmin:i_wmax]

    #---------------------------------------------
    # BINARY OPERATIONS
    def __mul__(self, n):
        # float multipication
        if type(n) == type(1.0):
            final_spec = Spectrum(wave=self.__wave,
                                  flux=n*self.__flux,
                                  var=(n**2)*self.__var)
            return final_spec
        # spectrum multiplication
        if type(n) == type(self):
            # global wavelength array
            final_wave = np.unique(np.concatenate([self.__wave,
                                                         n.wave()]))
            # fluxes of two specs
            flux1 = self.flux(final_wave)
            flux2 = n.flux(final_wave)
            final_flux = flux1*flux2
            # variances!
            var1 = self.var(final_wave)
            var2 = n.var(final_wave)
            final_var = (flux1**2)*var2+(flux2**2)*var1
            # return final object
            final_spec = Spectrum(wave=final_wave,
                                  flux=final_flux,
                                  var=final_var)
            return final_spec

    def __add__(self, n):
        # float multipication
        if type(n) == type(1.0):
            final_spec = Spectrum(wave=self.__wave,
                                  flux=n+self.__flux,
                                  var=self.__var)
            return final_spec
        # spectrum multiplication
        if type(n) == type(self):
            # global wavelength array
            final_wave = np.unique(np.concatenate([self.__wave,
                                                         n.wave()]))
            # fluxes of two specs
            flux1 = self.flux(final_wave)
            flux2 = n.flux(final_wave)
            final_flux = flux1+flux2
            # variances!
            var1 = self.var(final_wave)
            var2 = n.var(final_wave)
            final_var = var2+var1
            # return final object
            final_spec = Spectrum(wave=final_wave,
                                  flux=final_flux,
                                  var=final_var)
            return final_spec

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# DATA I/O FUNCTIONS
def load_spectrum(spec_fn, wmin=None, wmax=None, z_orig=0.0):
    try:
        spec_data = np.loadtxt(spec_fn)
    except:
        init_spec_data = []
        f1 = open(spec_fn, 'r')
        spec_lines = f1.readlines()
        f1.close()
        for line in spec_lines:
            try:
                new_data = [float(x) for x in line.split()]
                init_spec_data.append(new_data)
            except:
                continue
        spec_data = np.array(init_spec_data)
    spec_wave = spec_data[:,0]/(1.0+z_orig)
    spec_flux = spec_data[:,1]
    try:
        spec_var = spec_data[:,2]**2
    except:
        spec_var = np.ones(len(spec_wave), dtype = 'd')
    # set wavelength bounds
    if wmin == None:
        wmin = spec_wave[0]
    if wmax == None:
        wmax = spec_wave[-1]
    i_wmin = np.nonzero(spec_wave >= wmin)[0][0]
    i_wmax = np.nonzero(spec_wave <= wmax)[0][-1]
    return Spectrum(wave=spec_wave[i_wmin:i_wmax],
                    flux=spec_flux[i_wmin:i_wmax],
                    var=spec_var[i_wmin:i_wmax])

def write_spectrum(spectrum, spec_fn,
                   save_var = True):
    wave = spectrum.wave()
    flux = spectrum.flux()
    var = spectrum.var()
    sig = var**0.5
    nw = len(wave)
    f1 = open(spec_fn, 'w')
    for i in range(nw):
        if save_var:
            save_str = '%f %e %e' % (wave[i],
                                     flux[i],
                                     sig[i])
        else:
            save_str = '%f %e' % (wave[i],
                                  flux[i])
        print(save_str,file=f1)
    f1.close()
    return

"""
def load_filter(filt_fn):
    filt_data = np.loadtxt(filt_fn)
    filt_wave = filt_data[:,0]
    filt_tp   = filt_data[:,1]
    filt_var  = np.zeros(len(filt_wave), dtype = 'd')
    return Spectrum(wave=filt_wave,
                    flux=filt_tp,
                    var=filt_var)
"""

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# redshifting code!
def redshift(spec, z):
    new_spec = Spectrum(wave=spec.wave()*(1.0+z),
                        flux=spec.flux(),
                        var=spec.var())
    return new_spec

def deredshift(spec, z):
    return redshift(spec, -1.0*z)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# synthetic photometry code
def synphot(spec, filt, z=0.0):
    zspec = redshift(spec, z)
    synphot_spec = zspec*filt
    # sum it up!
    wave = synphot_spec.wave()
    sp = synphot_spec.flux()
    dwave = wave[1:]-wave[:-1]
    spave = 0.5*(sp[1:]+sp[:-1])
    return np.sum(dwave*spave)

def synphot_err(spec, filt, z=0.0):
    zspec = redshift(spec, z)
    alt_filt = Spectrum(wave = filt.wave(),
                        flux = filt.flux(),
                        var = 0.0*filt.var())
    synphot_spec = zspec*alt_filt
    # sum it up!
    wave = synphot_spec.wave()
    sp_var = synphot_spec.var()
    sp_var[np.nonzero(sp_var!=sp_var)[0]] = 0.0
    dwave = wave[1:]-wave[:-1]
    sp_var_ave = 0.5*(sp_var[1:]+sp_var[:-1])
    return np.sum((dwave**2)*sp_var_ave)**0.5

def mag(spec, filt, z=0.0, zp=0.0):
    return zp - 2.5*np.log10(synphot(spec, filt, z))

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# spectrum modification / combination / smoothing
def combine_spectra(blue_spec, red_spec,
                    xchan=1.0,
                    wsplit = 'mid'):
    # get all the values
    blue_wave = blue_spec.wave()
    blue_flux = blue_spec.flux()
    blue_var  = blue_spec.var()
    red_wave = red_spec.wave()
    red_flux = red_spec.flux()
    red_var  = red_spec.var()
    # figure out the split wavelength
    if wsplit == 'mid':
        wsplit = 0.5*(blue_wave.max()+red_wave.min())
    elif wsplit == 'blue_wmax':
        wsplit = blue_wave.max()
    elif wsplit == 'red_wmin':
        wsplit = red_wave.min()
    elif type(wsplit) == type(1.0):
        pass
    else:
        raise (ValueError, 'unrecognized wave split')
    # combine stuff
    blue_inds = np.nonzero(blue_wave <= wsplit)[0]
    red_inds = np.nonzero(red_wave > wsplit)[0]
    com_wave = np.concatenate([blue_wave[blue_inds],
                                  red_wave[red_inds]])
    com_flux = np.concatenate([xchan*blue_flux[blue_inds],
                                  red_flux[red_inds]])
    com_var = np.concatenate([(xchan**2)*blue_var[blue_inds],
                                  red_var[red_inds]])
    out_spec = Spectrum(wave = com_wave,
                        flux = com_flux,
                        var = com_var)
    return out_spec

def bin_spectrum(spec, nbin=2,
                 method = 'mean'):
    wave = spec.wave()
    flux = spec.flux()
    var = spec.var()
    nw = len(wave)
    new_nw = nw/nbin
    nf = new_nw*nbin
    if method == 'mean':
        bfunc = np.mean
    elif method == 'median':
        bfunc = np.median
    else:
        raise (ValueError)
    new_wave = bfunc(np.reshape(wave[:nf], [new_nw, nbin]), axis=1)
    new_flux = bfunc(np.reshape(flux[:nf], [new_nw, nbin]), axis=1)
    new_var  = bfunc(np.reshape(var[:nf], [new_nw, nbin]), axis=1)
    new_spec = Spectrum(wave=new_wave,
                        flux=new_flux,
                        var=new_var)
    return new_spec
def gaussian(x_array, sigma, center):
    return (np.exp(-((x_array - center)**2/(2.0*sigma**2)))
            /(2.0*np.pi*sigma**2)**0.5)

def smooth_spectrum(spec,
                    polydeg=3,
                    gauss_sig = 30.0):
    wave = spec.wave()
    flux = spec.flux()
    var  = spec.var()
    if len(var) != len(flux):
        var = np.ones(len(flux), dtype='d')
    # apply a rolling smoothing function
    # chi^2 weighted polynomial fit,
    xwin = int(gauss_sig/(wave[1]-wave[0]))
    nw = len(wave)
    smooth_flux = np.zeros(nw, dtype = 'd')
    for i in range(nw):
        weights = (
            gaussian(wave, gauss_sig, wave[i]) # gaussian weights
            /var) # variance weighting
        wsum = np.sum(weights)
        poly_chi2 = lambda p : np.sum(
            ((flux-np.polyval(p,wave))**2)*weights)/wsum
        p_guess = np.polyfit(wave[max(0,i-xwin):min(nw,i+xwin+1)],
                                flux[max(0,i-xwin):min(nw,i+xwin+1)],
                                polydeg)
        best_p = scipy.optimize.fmin(poly_chi2, p_guess, disp=0)
        smooth_flux[i] = np.polyval(best_p, wave)[i]
    # return smoothed spectrum
    smooth_spec = Spectrum(wave = wave,
                           flux = smooth_flux,
                           var = var)
    return smooth_spec

def velocity_convolve_spectrum(spec, vel):
    wave = spec.wave()
    flux = spec.flux()
    var  = spec.var()
    # set up saved flux and var
    nw = len(wave)
    out_flux = np.zeros(nw, dtype='d')
    out_var  = np.zeros(nw, dtype='d')
    # convolve it!
    for i in range(nw):
        curr_w = wave[i]
        curr_sig = curr_w*(vel/2.9979e5)
        init_weights = np.exp(-0.5*((wave-curr_w)/curr_sig)**2)
        weights = init_weights / np.sum(init_weights)
        out_flux[i] = np.sum(weights*flux)
        out_var[i] = np.sum(weights*var)
    # output spectrum
    out_spec = Spectrum(wave=wave,
                        flux=out_flux,
                        var=out_var)
    return out_spec

def wavelength_convolve_spectrum(spec, wconv):
    wave = spec.wave()
    flux = spec.flux()
    var  = spec.var()
    # set up saved flux and var
    nw = len(wave)
    out_flux = np.zeros(nw, dtype='d')
    out_var  = np.zeros(nw, dtype='d')
    # convolve it!
    for i in range(nw):
        curr_w = wave[i]
        curr_sig = wconv
        init_weights = np.exp(-0.5*((wave-curr_w)/curr_sig)**2)
        weights = init_weights / np.sum(init_weights)
        out_flux[i] = np.sum(weights*flux)
        out_var[i] = np.sum(weights*var)
    # output spectrum
    out_spec = Spectrum(wave=wave,
                        flux=out_flux,
                        var=out_var)
    return out_spec

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Savitsky-Golay smoothing
def SG_smooth(x, y, window, polydeg, good_inds=None):
    nx = len(x)
    good_mask = np.ones(nx)
    if good_inds != None:
        good_mask *= 0
        good_mask[good_inds] = 1
    smooth_y = np.zeros(nx, dtype='d')
    for i in range(nx):
        xi = x[i]
        fit_inds = np.nonzero((good_mask)*
                                 (x >= xi-window)*
                                 (x <= xi+window))[0]
        if len(fit_inds) == 0:
            smooth_y[i] = np.nan
            continue
        fit_x = x[fit_inds]
        fit_y = y[fit_inds]
        curr_fit = np.polyfit(fit_x, fit_y, polydeg)
        fitted_y = np.polyval(curr_fit, np.array([xi]))[0]
        smooth_y[i] = fitted_y
    return smooth_y

def SG_smooth_spec(in_spec, window, polydeg):
    wave = in_spec.wave()
    flux = in_spec.flux()
    var  = in_spec.var()
    out_flux = SG_smooth(wave, flux, window, polydeg)
    out_spec = Spectrum(wave=wave, flux=out_flux, var=var)
    return out_spec

# Extra spectral utilities
def convert_escma_fluxes_to_griz_mags(flux_array,
                                ):
    """
    Input MUST be 4x arrays of griz fluxes
    """
    obs_mags = -2.5*np.log10(flux_array/zp_fluxes)
    obs_fuJy = 10**(0.4*(23.9-obs_mags))
    return obs_mags.values, obs_fuJy.values

def rebin_spec(wave, specin, wavnew):

    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
    return obs.binflux

def rebin_a_spec(wave,flux,new_wave):

    wavenew = []
    fluxnew = []

    for counter,w in enumerate(wave):
        if wave[counter-1] ==w:
            fluxnew[-1]+=flux[counter]
        else:
            wavenew.append(w)
            fluxnew.append(flux[counter])
    return rebin_spec(np.array(wavenew),np.array(fluxnew),new_wave)

# Classes to deal with nebular lines
# Author: Médéric Boquien


class NebularLines(object):
    """Nebular lines templates.

    This class holds the data associated with the line templates

    """

    def __init__(self, metallicity, logU, name, wave, ratio):
        """Create a new nebular lines template

        Parameters
        ----------
        metallicity: float
            Gas phase metallicity
        logU: float
            Ionisation parameter
        name: array
            Name of each line
        wave: array
            Vector of the λ grid used in the templates [nm]
        ratio: array
            Line intensities relative to Hβ

        """

        self.metallicity = metallicity
        self.logU = logU
        self.name = name
        self.wave = wave
        self.ratio = ratio

class NebularContinuum(object):
    """Nebular lines templates.

    This class holds the data associated with the line templates

    """

    def __init__(self, metallicity, logU, wave, lumin):
        """Create a new nebular lines template

        Parameters
        ----------
        metallicity: float
            Gas phase metallicity
        logU: float
            Ionisation parameter
        wave: array
            Vector of the λ grid used in the templates [nm]
        lumin: array
            Luminosity density of the nebular continuum in Fλ

        """

        self.metallicity = metallicity
        self.logU = logU
        self.wave = wave
        self.lumin = lumin

  # Function to iterpolate SFHs onto the BC03 age grid
def interpolate_SFH(sfh,mtot):
    '''Function to iterpolate SFHs onto the BC03 age grid '''
    sfh['stellar_age'] = sfh.age.values[::-1]
    gb =sfh.groupby(pd.cut(sfh['stellar_age'],bins=np.concatenate([[0],10**(bc03_logt_float_array)/(1E+6)]))).agg(sum)

    return gb['m_formed'].values/mtot
