from spectral_utils import load_spectrum, Spectrum, redshift, synphot, rebin_a_spec, NebularLines, NebularContinuum
from what_the_flux import what_the_flux as wtf
import numpy as np
import pandas as pd
import os
from astropy.table import Table
import astropy.units as u
from scipy.stats import norm
from dust_extinction.parameter_averages import CCM89, F19
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
aura_dir = '/media/data3/wiseman/des/AURA/'
def phi_t_pl(t,tp,s,norm):
            '''Functional form of the delay time distribution'''
            return ((0*t)*(t<tp))+((norm*(t**s))*(t>tp))


class SynSpec():
    def __init__(self,root_dir=aura_dir,template_obj_list =None,neb=False,library='BC03',template_dir=None):
        self.root_dir = root_dir
        # something
        if not template_obj_list:
            self.template_obj_list = self._get_templates(library=library,template_dir=template_dir)
        else:
            self.template_obj_list = template_obj_list
        self.ntemplates=len(self.template_obj_list)
        self.filt_dir = self.root_dir+'filters/'
        self.filt_obj_list = self._get_filters()
        self.nfilt=len(self.filt_obj_list)
        self.library = library
        if neb:
            self._prep_neb()

    def _get_templates(self,library,template_dir='/media/data1/childress/des/galaxy_sfh_fitting/bc03_ssp_templates/',ntemp=None,logt_list=None):
        #bc03_dir = '/media/data1/childress/des/galaxy_sfh_fitting/bc03_ssp_templates/'
        template_obj_list = []

        if library == 'BC03':
            for i in range(ntemp):
                bc03_fn = '%sbc03_chabrier_z02_%s.spec' % (template_dir, logt_list[i])
                new_template_spec = load_spectrum(bc03_fn)
                template_obj_list.append(new_template_spec)
        elif library == 'PEGASE':
            templates = pd.read_hdf(template_dir + 'templates.h5')
            templates.drop(
                ['m_gal', 'm_star', 'm_wd', 'm_nsbh', 'm_substellar', 'm_gas', 'z_ism', 'z_stars_mass', 'z_stars_bl',
                 'l_bol', 'od_v', 'l_dust_l_bol',
                 'sfr', 'phot_lyman', 'rate_snii', 'rate_snia', 'age_star_mass', 'age_star_lbol'],
                axis=1, inplace=True)
            templates.set_index('time', drop=True, inplace=True)
            templates.columns = templates.columns.astype(float)
            templates = templates.T
            templates.sort_index(inplace=True)
            templates = templates.T
            for i in templates.index:
                template_obj_list.append(
                    Spectrum(wave=templates.columns, flux=templates.loc[i].values, var=np.ones_like(templates.loc[i])))
        return template_obj_list

    def _get_filters(self,):
        filt_obj_list = [
            load_spectrum(self.filt_dir+'decam_g.dat'),
            load_spectrum(self.filt_dir+'decam_r.dat'),
            load_spectrum(self.filt_dir+'decam_i.dat'),
            load_spectrum(self.filt_dir+'decam_z.dat'),
        ]
        return filt_obj_list
    def _prep_neb(self,Z=0.02):
        import scipy.constants as cst

        nebular_dir = os.path.join('/media/data3/wiseman/des/cigale/cigale-v2020/database_builder/', 'nebular/')
        print("Importing {}...".format(nebular_dir + 'lines.dat'))
        lines = np.genfromtxt(nebular_dir + 'lines.dat')

        tmp = Table.read(nebular_dir + 'line_wavelengths.dat', format='ascii')
        wave_lines = tmp['col1'].data
        name_lines = tmp['col2'].data

        print("Importing {}...".format(nebular_dir + 'continuum.dat'))
        cont = np.genfromtxt(nebular_dir + 'continuum.dat')

        # Convert wavelength from Å to nm
        wave_lines *= 0.1
        wave_cont = cont[:3729, 0] * 0.1

        # Get the list of metallicities
        metallicities = np.unique(lines[:, 1])

        # Keep only the fluxes
        lines = lines[:, 2:]
        cont = cont[:, 1:]

        # We select only models with ne=100. Other values could be included later
        lines = lines[:, 1::3]
        cont = cont[:, 1::3]

        # Convert lines to W and to a linear scale
        lines = 10**(lines-7)

        # Convert continuum to W/nm
        cont *= np.tile(1e-7 * cst.c * 1e9 / wave_cont**2,
                        metallicities.size)[:, np.newaxis]

        self.models_lines = {}
        lines_width=300
        # Import lines
        for idx, metallicity in enumerate(metallicities):
            spectra = lines[idx::6, :]
            self.models_lines[metallicity] = {}
            for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                      spectra.T):
                lines_obj = NebularLines(metallicity, logU, name_lines,
                                                 wave_lines, spectrum)
                new_wave = np.array([])
                for line_wave in lines_obj.wave:
                    width = line_wave * lines_width * 1e3 / cst.c
                    new_wave = np.concatenate((new_wave,
                                            np.linspace(line_wave - 3. * width,
                                                        line_wave + 3. * width,
                                                        9)))

                new_wave.sort()
                new_flux = np.zeros_like(new_wave)
                for line_flux, line_wave in zip(lines_obj.ratio, lines_obj.wave):
                    width = line_wave * lines_width * 1e3 / cst.c
                    new_flux += (line_flux * np.exp(- 4. * np.log(2.) *
                                (new_wave - line_wave) ** 2. / (width * width)) /
                                (width * np.sqrt(np.pi / np.log(2.)) / 2.))
                lines_obj.wave = new_wave
                lines_obj.ratio = new_flux
                self.models_lines[metallicity][logU]=lines_obj
            # Import continuum

        '''self.models_cont = {}
        for idx, metallicity in enumerate(metallicities):
            spectra = cont[3729 * idx: 3729 * (idx+1), :]
            for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                      spectra.T):
                self.models_cont[metallicity][logU] = NebularContinuum(metallicity, logU, wave_cont,
                                                    spectrum)'''

        # load the Lyman Continuum flux from the templates
        ssp_fn = os.path.join(self.root_dir,'BC03/bc03/models/Padova1994/chabrier/')
        ssp_vals = Table.read(ssp_fn+'bc2003_hr_m62_chab_ssp.3color',format='ascii')
        nLy = ssp_vals['col6']
        self.nLy = np.array(nLy)

    def get_ssp_fluxes(self,z,filters ='obs',frame='obs',):
        flux_array = np.zeros([ntemp, nfilt], dtype='d')
        if frame=='obs':
            for i in range(self.ntemp):
                curr_temp = self.template_obj_list[i]
                if filters=='Obs':
                    for j in range(self.nfilt):
                        curr_filt = self.filt_obj_list[j]
                        # calculate flux, save it to the array
                        curr_flux = synphot(curr_temp, curr_filt, z=z)
                        flux_array[i,j] = curr_flux
                else:
                    for j in range(len(filters)):
                        curr_filt = filters[j]
                        # calculate flux, save it to the array
                        curr_flux = synphot(curr_temp, curr_filt, z=z)
                        flux_array[i,j] = curr_flux
        return flux_array

    def get_spec_fluxes(self,spec_list,z,filters ='obs',frame='obs',):
        ntemp = len(spec_list)
        if filters=='obs':
            nfilt=self.nfilt
        else:
            nfilt = len(filters)
        flux_array = np.zeros([ntemp, nfilt], dtype='d')
        if frame=='obs':
            for i in range(len(spec_list)):
                curr_temp = spec_list[i]
                if filters=='obs':
                    for j in range(self.nfilt):
                        curr_filt = self.filt_obj_list[j]
                        # calculate flux, save it to the array
                        #print('Mangling with DES filters',curr_temp)
                        curr_flux = synphot(curr_temp, curr_filt, z=z)
                        flux_array[i,j] = curr_flux
                else:
                    for j in range(len(filters)):
                        curr_filt = filters[j]
                        # calculate flux, save it to the array
                        #print('Mangling with user defined filters',len(curr_temp.wave()),len(curr_temp.flux()))
                        curr_flux = synphot(curr_temp, curr_filt, z=z)
                        flux_array[i,j] = curr_flux
        return flux_array



    def redden_a_combined_spec(self,wave,flux,law='F19',Av=0,Rv=3.1,B=2,delta=0):
        '''Reddens a spectrum according to an extinction law. Default is to return an unreddened spectrum'''
        if law=='F19':
            ext_model = F19(Rv=Rv)
            #print(len(wave))
            wave_inv_microns = 1/(wave/1E+4) /u.micron
            #print(len(wave_inv_microns))
            text_model = F19()
            lims =( wave_inv_microns>text_model.x_range[0]/u.micron)&( wave_inv_microns<text_model.x_range[1]/u.micron)

            ext_F19 = ext_model(wave_inv_microns[lims])
            #print(len(wave_inv_microns),len(ext_F19),len(lims),len(flux))
            return wave[lims],flux[lims]*10**(-0.4*ext_F19*Av)
        if law=='CCM89':
            ext_model = CCM89(Rv=Rv)
            #print(len(wave))
            try:
                wave_inv_microns = 1/(wave.values/1E+4) /u.micron
            except:
                wave_inv_microns = 1/(wave/1E+4)/u.micron
            #print(len(wave_inv_microns))
            text_model = CCM89()
            lims =( wave_inv_microns>text_model.x_range[0]/u.micron)&( wave_inv_microns<text_model.x_range[1]/u.micron)

            ext_CCM89 = ext_model(wave_inv_microns[lims])
            #print(len(wave_inv_microns),len(ext_CCM89),len(lims),len(flux))
            return wave[lims],flux[lims]*10**(-0.4*ext_CCM89*Av)
        elif law=='SBL18':
            att_model = SBL18(Av=Av,slope=delta,ampl=B)
            x_range = att_model.x_range
            lims = (wave > x_range[0]*1E+4)&(wave < x_range[1]*1E+4)
            wave = wave*u.AA
            att = att_model(wave[lims])
            return wave[lims].value,flux[lims]*10**(-0.4*att)
    def calculate_colour(self,spec_list,flt1='UX',flt2='RJ'):
        filter1 = load_spectrum(self.filt_dir+'%s_B90.dat'%flt1)
        filter2 = load_spectrum(self.filt_dir+'%s.dat'%flt2)
        #print('Loaded filters')
        fluxes = self.get_spec_fluxes(spec_list,z=0,filters=[filter1,filter2])
        #print('Got fluxes for each band: ',fluxes)
        vega_zps = {
            'UX': 417.5e-11,
            'RJ': 217.7e-11
        }
        mag_1 = -2.5 * np.log10(fluxes[:, 0] / vega_zps[flt1])
        mag_2 = -2.5 * np.log10(fluxes[:, 1] / vega_zps[flt2])
        return mag_1 - mag_2

    def calculate_colour_wtf(self, spec_list, flt1='U', flt2='R'):
        filter1 = load_spectrum(self.filt_dir + 'Bessell%s.dat' % flt1)
        filter2 = load_spectrum(self.filt_dir + 'Bessell%s.dat' % flt2)
        band1 = wtf.Band_Vega(filter1.wave(), filter1.flux() * u.erg / u.s / u.AA)
        band2 = wtf.Band_Vega(filter2.wave(), filter2.flux() * u.erg / u.s / u.AA)
        colours = []
        for s in spec_list:
            try:
                spec = wtf.Spectrum(s.wave().values * u.AA, s.flux() * u.erg / u.AA / u.s / u.cm / u.cm)
            except:
                spec = wtf.Spectrum(s.wave() * u.AA, s.flux() * u.erg / u.AA / u.s / u.cm / u.cm)
            mag1 = -2.5 * np.log10(spec.bandflux(band1).value / band1.zpFlux().value)
            mag2 = -2.5 * np.log10(spec.bandflux(band2).value / band2.zpFlux().value)
            colours.append(mag1 - mag2)
        return colours

    def get_bands_wtf(self,spec_list,band_dict):
        colours ={}
        absmag_corr = 1/((10*u.pc.to(u.cm))**2)
        for f,ftype in band_dict.items():
            colours[f] =[]
            filter = load_spectrum(self.filt_dir+'%s.dat'%f)
            if ftype=='Vega':
                wtf_filter =wtf.Band_Vega(filter.wave(), filter.flux() )
            elif ftype=='AB':
                wtf_filter =wtf.Band_AB(filter.wave(), filter.flux() )
            for s in spec_list:
                try:
                    spec = wtf.Spectrum(s.wave().values * u.AA, s.flux()*absmag_corr * u.erg / u.AA / u.s )
                except:
                    spec = wtf.Spectrum(s.wave() * u.AA, s.flux()*absmag_corr * u.erg / u.AA / u.s )
                colours[f].append(-2.5*np.log10(spec.bandflux(wtf_filter).value/wtf_filter.zpFlux().value))
        return colours
    def synphot_model_spectra_pw(self,sfh_coeffs,):


        #model_fluxes = sfh_coeffs*ssp_fluxes
        model_fluxes = np.array(
            np.matrix(sfh_coeffs)*np.matrix([s.flux() for s in self.template_obj_list]),dtype='object')

        return model_fluxes

    def synphot_model_emlines(self,sfh_coeffs,Z=0.02,logU=-4):
        em_lines = self.models_lines[Z][logU]
        em_waves = em_lines.wave

        #lHb_arr = 10**(self.nLy)#*4.757E-13 * 1E-7 # luminosity of H beta in Watts
        #lHb_arr/= 3.12e7 # convert to BC03 units
        nLy_arr = 10**self.nLy

        lHbs = nLy_arr * sfh_coeffs
        line_strengths = np.multiply(np.matrix(lHbs),np.matrix(em_lines.ratio).T)

        em_lums = np.array(np.sum(line_strengths,axis=1))

        return em_waves,em_lums[:,0]/3.826e27

    def calculate_model_fluxes_pw(self,sfh_coeffs,z,dust=None,neb=False,logU=-2,mtot=1E+10):
        #print('Combining the weighted SSPs for this SFH')
        model_spec = self.synphot_model_spectra_pw(sfh_coeffs)[0]
        print(np.log10(np.max(model_spec)))
        wave = self.template_obj_list[0].wave()
        model_spec = Spectrum(wave=wave,
                    flux=model_spec,
                    var=np.ones_like(model_spec))

        if neb:
            model_neb_wave,model_neb_flux= self.synphot_model_emlines(sfh_coeffs,logU=logU)
            #print(model_neb_wave)

            model_neb_flux_rebinned = rebin_a_spec(model_neb_wave*10,model_neb_flux/10,model_spec.wave())
            #self.model_neb = Spectrum(wave=model_spec.wave(),flux=model_neb_flux_rebinned,var = np.ones_like(model_neb_flux_rebinned))
            model_spec = Spectrum(wave=model_spec.wave(),flux = model_spec.flux() + model_neb_flux_rebinned,var = model_spec.var())
        #print('Going to redden my model spectrum')
        #self.model_spec = model_spec
        if not dust:

            model_spec_reddened = model_spec
        else:
            #print('Reddening with this dust: ',dust)
            try:
                wave,flux = self.redden_a_combined_spec(model_spec.wave(),model_spec.flux(),law=dust['law'],Av=dust['Av'],Rv=dust['Rv'],delta=dust['delta'])
            except:
                wave, flux = self.redden_a_combined_spec(model_spec.wave(), model_spec.flux(), law=dust['law'],
                                                         Av=dust['Av'], Rv=dust['Rv'], delta=dust['delta'])

            var = np.ones_like(wave)
            model_spec_reddened=Spectrum(wave=wave,flux=flux,var=var)
        self.model_spec = model_spec
        #print('I reddened things, they look like this: ',model_spec_reddened)
        #f,ax=plt.subplots()
        #ax.step(model_spec_reddened.wave(),model_spec_reddened.flux())
        #ax.set_xlim(2500,12000)
        #print('Going go calculate restframe colour')
        if self.library =='BC03':
            bc03_flux_conv_factor =  3.12e7
        else:
            bc03_flux_conv_factor = 1
        model_spec_reddened =Spectrum(wave=model_spec_reddened.wave(),
                                      flux=model_spec_reddened.flux()*mtot/(bc03_flux_conv_factor),
                                      var=np.ones_like(model_spec_reddened.wave()))
        print(np.log10(mtot),np.log10(np.max(model_spec_reddened.flux())),np.log10(np.mean(model_spec_reddened.flux())))
        colour = self.calculate_colour_wtf([model_spec_reddened])
        colours = self.get_bands_wtf([model_spec_reddened],band_dict={'Bessell%s'%b:'Vega' for b in ['U','B','V','R','I']})
        #print('Here is the colour: ',colour)
        #print('Going go calculate observed flux with this',model_spec_reddened)
        des_fluxes = self.get_spec_fluxes([model_spec_reddened],z)/(1+z) #extra 1+z for flux densities
        return colour, des_fluxes, colours
