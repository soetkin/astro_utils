import numpy as np
from dust_extinction.parameter_averages import F19
import astropy.units as u
from datetime import datetime
from astropy.io import fits
from scipy import interpolate
from astro_utils import constants as const
from astro_utils import input_output as inout
from astro_utils.fit_funcs import fit_functions as fit

#Hey

def convert_mag_to_flux(mag, flux_zeropoint, mag_err=None, mag0=0):
    """
    convert magnitudes into fluxes: unit => [erg/s/cm2/AA]
    - reference magnitude m0: zero for VEGA magnitudes
    - error from error propagation: d/dmag(10^mag) = 10^mag * ln(10)
    - tested with IvS repository, which agrees on errors and AB magnitudes
    - input: magnitude, flux zeropoint in [erg/s/cm^2/AA], (magnitude error)
    - returns: flux (and flux error) in [erg/s/cm^2/AA]
    """

    flux = flux_zeropoint * 10**(-(mag - mag0)/2.5)  # [erg/s/cm2/AA]

    # only compute and return a flux error if a magnitude error is given
    if mag_err is None:
        return flux
    if mag_err is not None:
        flux_err = flux * np.log(10)/2.5 * mag_err
        return flux, flux_err


def prep_sed(wave, flux, radius_pc, distance_pc, E_BV=0.08, R_V=3.1,
             w_min=1100, w_max=33000):
    """
    prepare a model SED to compare it with observations
    - cut in wavelength range
    - apply interstellar reddening
    - scale for distance (in pc) and radius (in pc) of the star
    - for now: reddening Fitzpatrick+2019 law, implemented up to 3.3 micron
    - todo: implement other reddening laws, also up to higher wavelength
    - input: wavelength, flux of the model that is reddened, radius of the star
             in [pc], distance to the star in [pc] (and possibly E(B-V), R(V),
             wavelength range to cut model in)
    - returns: wavelength, flux of model; cut, reddened, scaled for Rstar and d
    """
    # cut in wl range where reddening law is implemented
    wave_cut, flux_cut = inout.cut_specrange(wave, flux, w_min, w_max)

    # scale for radius and distance
    f_obs = 4 * np.pi * radius_pc**2 / distance_pc**2 * flux_cut

    # redden the model
    ext = F19(Rv=R_V)
    f_red = f_obs * ext.extinguish(0.0001*wave_cut*u.micron, Ebv=E_BV)

    return wave_cut, f_red


def prep_obs_vot(infile, fnames, fnames_vizier, plot_flag=False):
    """
    read in photmetric information from input file
    - selects desired filters (compatible with filters module)
    - if plot_flag = True: make a plot of the input and selected photometry
    - for now: expects vot table from vizier
    - todo: expanded to different kind of input formats
    - input: photometry input file (for now: vot table from vizier), filter
             names recognized by filters module, vizier filter names, plot flag
    - returns: arrays of wavelengths, fluxes and errors in [erg/s/cm^2/A] in
               selected filters
    """

    # converts fluxes from [Jy] to [erg/s/cm^2/AA]
    v_tab = inout.read_vot_vizier(infile)
    waves, fluxes, flux_errs, filters = v_tab[0], v_tab[1], v_tab[2], v_tab[3]

    # prepare arrays containing selected observations
    wave_obs, flux_obs, flux_errs_obs = [], [], []
    for f in range(len(fnames)):
        # find vizier filter that corresponds to desired filter
        idx = np.where(filters == fnames_vizier[f])[0][0]

        # select corresponding wave, flux, err values and append to return list
        wave_obs.append(waves[idx])
        flux_obs.append(fluxes[idx])
        flux_errs_obs.append(flux_errs[idx])

    if plot_flag is True:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.errorbar(waves, fluxes, yerr=flux_errs, color='dimgray',
                    marker='+', ls='None', lw=0.7, label='all obs')
        ax.errorbar(wave_obs, flux_obs, yerr=flux_errs_obs, color='crimson',
                    marker='.', ls='None', label='obs selected for fit')
        ax.plot(wave_obs, flux_obs, color='crimson', alpha=0.3, ls=':', lw=0.5)

        ax.legend()
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel(r'Flux [erg/s/cm$^2$/$\AA$]')
        plt.show()

    return wave_obs, flux_obs, flux_errs_obs


def conv_filter(wave_in, flux_in, wave_filter, flux_filter):
    """
    convolve a model SED with a given filter and compute the flux in the filter
    - cut the model SED according to the wavelength range of the filter
    - interpolate the filter so that filter and SED have same wavelength steps
    - convolves model with filter and integrates flux [erg/s/cm^2/AA]
    - expects normalized filters
    - input: wavelength, flux of model; wavelength, normalized flux of filter
    - returns: wave, flux of filter * model, integrated flux [erg/s/cm^2/AA]
    """
    # cut the input SED at the borders of the filter
    wave_cut, flux_cut = inout.cut_specrange(wave_in, flux_in, wave_filter[0],
                                             wave_filter[-1])

    # interpolate so that they have the same stepsize
    s = interpolate.interp1d(wave_filter, flux_filter, 1)
    flux_f_interp = s(wave_cut)

    # multiply SED with filter and compute flux
    flux_fil = flux_cut * flux_f_interp
    int_flux_filter = np.sum(flux_fil) / np.sum(flux_f_interp)  # erg/s/cm^2/AA

    return wave_cut, flux_fil, int_flux_filter


def flux_from_mag_header(infile):
    """
    convenience function for MUSE data where HST magnitudes are in fits header
    - read in F225W, F336W and F814W magnitudes from fits header
    - convert magnitudes to fluxes [erg/s/cm^2/AA]
    - errors and n(measurements) only given for F225W and F336W
    - compute errors from c_factors (low number of measurements)
    - for F814W: take relative error from F225W measurements
    - input: MUSE fits file (prepared by extraction script)
    - returns: fluxes and errors in F225W, F336W, F814W
    """
    # definitions for HST
    c2, c3, c4 = 0.7978845608, 0.8862269255, 0.9213177319
    c5, c6, c7 = 0.9399856030, 0.9515328619, 0.9593687891

    #                 0,      1,  2,  3,  4,  5 , 6, 7
    c_factors = [np.nan, np.nan, c2, c3, c4, c5, c6, c7]

    # data from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php
    f0_225W = 4.23804*10**(-9)  # (erg/cm2/s/A)
    f0_336W = 3.26021*10**(-9)  # (erg/cm2/s/A)
    f0_814W = 1.14324*10**(-9)  # (erg/cm2/s/A)

    head = fits.getheader(infile)
    # read in magnitudes from fits header, in this case HST magnitudes
    F225W = head['HIERARCH SPECTRUM F225W']
    F225W_err = head['HIERARCH SPECTRUM E225W']
    F225W_n = head['HIERARCH SPECTRUM N225W']

    F336W = head['HIERARCH SPECTRUM F336W']
    F336W_err = head['HIERARCH SPECTRUM E336W']
    F336W_n = head['HIERARCH SPECTRUM N336W']

    F814W = head['HIERARCH SPECTRUM F814W']

    # convert magnitudes into fluxes, UNIT: [erg/s/cm2/AA]
    f_225W, f_225W_err = convert_mag_to_flux(F225W, f0_225W, mag_err=F225W_err)
    f_336W, f_336W_err = convert_mag_to_flux(F336W, f0_336W, mag_err=F336W_err)
    f_814W = convert_mag_to_flux(F814W, f0_814W, mag_err=None)

    # take into account low number of images from Milone for error estimate
    f_225W_err = f_225W_err / c_factors[F225W_n]
    f_336W_err = f_336W_err / c_factors[F336W_n]

    # f814w = relative error from f225w error
    f_814W_err = f_225W_err / f_225W * f_814W  # erg/s/cm2/AA

    return f_225W, f_225W_err, f_336W, f_336W_err, f_814W, f_814W_err


def fit_sed(obs_flux, obs_flux_errs, dist, E_BV, R_V,
            waves_filter, fluxes_filter,
            teff_range, logg_range, radi_range, m_path,
            outfname):
    """
    fit observations with model SED using a chi^2 comparison
    - for now: 3 fitting parameters: teff, logg, radius (three for loops)
    - loop over teff and logg and selects corresponding tlusty model
    - loop over radius, scale model for (fixed) distance & extinction, radius
    - for each specified filter: convolve model with filter and compute flux
    - compute chi^2 between observed fluxes and computed model fluxes
    - write output file containing teff, logg, radius, dof and chi^2
    - input: observed fluxes with errors, distance [pc], E(B-V), R(V),
             wavelength and transmission of filters to consider,
             ranges of fitting parameters (teff, logg, radius), path to models,
             name of output file
    - returns: output file containing line for each model (teff, logg, radius,
               (global)dof, chi^2 of the respective model)
    """

    # outfile
    outf = open(outfname, 'w')
    outf.write('teff,logg,radius,dof,chi2' + '\n')

    # free parameters = 3
    # dof = len(obs_flux)  - n_free_params
    dof = len(obs_flux) - 3

    for t, teff in enumerate(teff_range):
        for g, logg in enumerate(logg_range):
            # tlusty file name
            model = 'BG' + str(teff) + 'g' + str(logg) + 'v2_sed.fits'

            # read in model (eddington flux at the stellar surface)
            wave_m, flux_m = inout.read_tlusty_fits(m_path + model,
                                                    fluxtype='fluxspec')
            for r, radius in enumerate(radi_range):
                # scale for radius, distance and extinction
                wave_c, f_red = prep_sed(wave_m, flux_m, radius, dist,
                                         E_BV, R_V,
                                         w_min=waves_filter[0][0],
                                         w_max=waves_filter[-1][-1])

                fvals_model = []
                # measure the flux in the respective filter bands
                for f in range(len(waves_filter)):
                    wave_f, flux_f = waves_filter[f], fluxes_filter[f]

                    (wave_f, flux_f,
                     int_flux_f) = conv_filter(wave_c, f_red,
                                               wave_f, flux_f)

                    fvals_model.append(int_flux_f)

                # compute chi2
                chi2 = fit.compute_chi2(obs_flux, fvals_model,
                                        obs_flux_errs)

                # write in outfile
                line = ('{}'.format(teff) + ',' +
                        '{:.2f}'.format(logg/100) + ',' +
                        '{}'.format(radius / const.Rsun_pc) + ',' +
                        '{}'.format(dof) + ',' +
                        '{:.8}'.format(chi2) + '\n')
                outf.write(line)

                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      '  ' + line)

    outf.close()
