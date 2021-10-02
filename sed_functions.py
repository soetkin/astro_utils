import numpy as np
from dust_extinction.parameter_averages import F19
import astropy.units as u
from astropy.io import fits
from scipy import interpolate
from julia_utils import input_output as inout


# definitions
c2, c3, c4 = 0.7978845608, 0.8862269255, 0.9213177319
c5, c6, c7 = 0.9399856030, 0.9515328619, 0.9593687891

#                 0,      1,  2,  3,  4,  5 , 6, 7
c_factors = [np.nan, np.nan, c2, c3, c4, c5, c6, c7]

# data from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php
f0_225W = 4.23804*10**(-9)  # (erg/cm2/s/A)
f0_336W = 3.26021*10**(-9)  # (erg/cm2/s/A)
f0_814W = 1.14324*10**(-9)  # (erg/cm2/s/A)


def convert_mag_to_flux(mag, flux_zeropoint, mag_err=None, mag0=0):
    # convert magnitudes into fluxes: unit => [erg/s/cm2/AA]
    # reference magnitude m0: zero for VEGA magnitudes
    # error from error propagation: d/dmag(10^mag) = 10^mag * ln(10)
    # tested with IvS repository, which agrees on errors and AB magnitudes
    flux = flux_zeropoint * 10**(-(mag - mag0)/2.5)  # [erg/s/cm2/AA]

    # only compute and return a flux error if a magnitude error is given
    if mag_err is None:
        return flux
    if mag_err is not None:
        flux_err = flux * np.log(10)/2.5 * mag_err
        return flux, flux_err


def prep_sed(wave, flux, radius, distance, E_BV=0.08, R_V=3.1):
    # cut in wl range where reddening law is implemented
    wave_cut, flux_cut = inout.cut_specrange(wave, flux, 1150, 9900)

    # scale for radius and distance
    f_obs = 4 * np.pi * radius**2 / distance**2 * flux_cut

    # redden the model
    ext = F19(Rv=R_V)
    f_red = f_obs * ext.extinguish(0.0001*wave_cut*u.micron, Ebv=E_BV)

    return wave_cut, f_red


def conv_filter(wave_in, flux_in, wave_filter, flux_filter):
    # cut the input spectrum / sed at the borders of the filter
    wave_cut, flux_cut = inout.cut_specrange(wave_in, flux_in, wave_filter[0],
                                             wave_filter[-1])

    # interpolate so that they have the same stepsize
    s = interpolate.interp1d(wave_filter, flux_filter, 1)
    flux_f_interp = s(wave_cut)

    # multiply SED with filter
    flux_fil = flux_cut * flux_f_interp
    int_flux_filter = np.sum(flux_fil) / np.sum(flux_f_interp)

    return wave_cut, flux_fil, int_flux_filter


def flux_from_mag_header(infile):
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
    print(F225W_n, c_factors[F225W_n])
    print(F336W_n, c_factors[F336W_n])
    f_225W_err = f_225W_err / c_factors[F225W_n]
    f_336W_err = f_336W_err / c_factors[F336W_n]

    # f814w = relative error from f225w error
    f_814W_err = f_225W_err / f_225W * f_814W  # erg/s/cm2/AA

    return f_225W, f_225W_err, f_336W, f_336W_err, f_814W, f_814W_err
