import numpy as np
import julia_utils.spec_functions as spec
import numpy.polynomial.polynomial as poly
from matplotlib.patches import Rectangle
import pandas as pd
# from scipy.integrate import simps


def measure_EW(wave, flux, wl_cent, line_label, snr, ax='None', vrad=150.,
               fit_wind=50., ew_wind=10., plot_wind=100.):
    """
    Measures the equivalent width of a certain spectral line given by wl_cent
    in a window ew_wind around the line.
    The spectrum is locally renormalized in a region around fit_wind.
    If ax is not None, then the spectrum is plotted in plot_wind.
    """
    # shift the lines to radial velocity (in this case: of SMC)
    wave_vrad = spec.doppler_shift(wave, vrad)

    # cut wavelength window in which a new continuum is fitted
    wl_n_s = wl_cent - fit_wind
    wl_n_e = wl_cent + fit_wind
    norm_range = ((wave_vrad > wl_n_s) & (wave_vrad < wl_n_e))
    wave_n = wave_vrad[norm_range]
    flux_n = flux[norm_range]

    # renormalize the spectrum locally - fit a line through beginning and end
    # take the medium of the last x values
    xval = 10
    x = [np.mean(wave_n[0:xval]), np.mean(wave_n[-xval:])]
    y = [np.nanmedian(flux_n[0:xval]), np.nanmedian(flux_n[-xval:])]

    coefs = poly.polyfit(x, y, 1)
    norm = poly.polyval(wave_n, coefs)

    # devide flux by that line
    norm_flux = flux_n / norm

    # only consider the selected wavelength window for measuring EW
    wl_cut_s = wl_cent - ew_wind
    wl_cut_e = wl_cent + ew_wind

    spec_range = ((wave_n > wl_cut_s) & (wave_n < wl_cut_e))
    wave_cut = wave_n[spec_range]
    flux_cut = norm_flux[spec_range]

    # measure the EW width by integrating over the selected window
    ew, err = spec.measure_equiwidth(wave_cut, flux_cut, snr)

    # if axis is not none: plot the (renormalized) spectrum
    if ax != 'None':
        # plot the whole spectral region
        ax.plot(wave_vrad, flux, color='k', alpha=0.5, linestyle=':',
                label='Input spectrum')

        # plot the region determined to fit normalization
        ax.add_patch(Rectangle((wl_n_s, -5), 2*fit_wind, 10., color='darkblue',
                               alpha=0.1))

        # plot the line that was fitted through the continuum
        ax.plot(wave_n, norm, alpha=0.8, linestyle='--', label='continuum')

        # plot the newly normalized wavelenth range
        ax.plot(wave_n, norm_flux, alpha=0.8, label='Re-normalized', color='k')

        # plot the newly normalized wavelenth range
        ax.plot(wave_cut, flux_cut, alpha=0.8, label='EW measure', color='C1')

        # set x and y limits
        ax.set_xlim([wl_cent - plot_wind, wl_cent + plot_wind])
        ax.set_ylim([0.7, 1.2])

        # plot a line at y = 1
        ax.axhline(y=1, alpha=0.8, color='k')
        ax.grid(b=True, which='major', linestyle=':', alpha=0.7)

        text = ('EW: (' + '{:.2f}'.format(ew) + r' $\pm$ ' +
                '{:.2f}'.format(err) + r') $\AA$')
        ax.annotate(text, (wl_cent+50, 0.8))

        ax.set_title('Equivalent Width of ' + line_label, fontsize=13)

        ax.legend(loc='lower left')

        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel(r'Normalized flux')

    return [ew, err]


def prep_herm(line, infile='None'):
    """
    Prepares the calibration from the standard stars. Expects a file containing
    at least the following three columns:
    index       contains numbers that correspond to SpTs
    ew_***      equivalent width of a certain line ***
    err_***     equivalent width error of a certain line ***
    """

    if infile == 'None':
        # path to input file containing HERMES classification data
        hermesfile = ('/lhome/julia/data/NGC330/MUSE/specclass_Ostars_Atlas/' +
                      'spec_class/measured_ews_hermesatlas.csv')

    # read in the data from the input file
    hermes_tab = pd.read_csv(hermesfile)

    lim = 0.1  # limit to which the measured values are considered significant

    # create array with numbers corresponding to SpT
    vals = np.array(hermes_tab['index'])

    # get the measured ew values and ew errs for standard stars from table
    line_key, err_key = 'ew_' + str(line), 'err_' + str(line)
    std_ew, std_ewerr = hermes_tab[line_key], hermes_tab[err_key]

    # only use the EWs in the fit that are significantly far away from 0 + lim
    ew_index = (std_ew > lim)
    use_vals = vals[ew_index]
    use_ews, use_ewerrs = std_ew[ew_index], std_ewerr[ew_index]

    # take erros on measured ews as weights when fitting polynomial
    weight = 1 / np.array(use_ewerrs)
    coefs = poly.polyfit(use_vals, use_ews, 2, w=weight)

    return vals, use_vals, std_ew, std_ewerr, use_ews, use_ewerrs, coefs


def get_spectype(fit_vals, fit, meas_ew, meas_ew_err):
    """
    Converts the measured ew and ew error into a number that corresponds to a
    spectral type. Requires a polynomial fit in spectral type - ew from
    standard stars as well as measured ew and ew err.
    """
    # make an array of equal size than the fitted array
    ew_val_array = np.full(len(fit), meas_ew)
    ew_u_err_array = np.full(len(fit), meas_ew+meas_ew_err)
    ew_l_err_array = np.full(len(fit), meas_ew-meas_ew_err)

    used_vals_spaced = np.array(fit_vals)

    # get the intersection between the two curves
    idx = np.argwhere(np.diff(np.sign(fit - ew_val_array))).flatten()
    specval = used_vals_spaced[idx]

    # get intersection with EW + EWERR
    idx_u = np.argwhere(np.diff(np.sign(fit - ew_u_err_array))).flatten()
    specval_u = used_vals_spaced[idx_u]

    # get intersection with EW - EWERR
    idx_l = np.argwhere(np.diff(np.sign(fit - ew_l_err_array))).flatten()
    specval_l = used_vals_spaced[idx_l]

    # in case the upper or lower limit is out of bounds
    if len(specval_u) == 0:
        specval_u = [0, -3]
    if len(specval_l) == 0:
        specval_l = [10, 0]
    if len(specval) == 0:
        specval = [10, 0]

    return specval, specval_u, specval_l


def check_range(use_vals, coefs, meas_ew, meas_ewerr, ax='None'):
    """
    Wrapper function that converts the measured equivalent width into a number
    that corresponds to spectral type using the standard star calibration.
    If ax is not None: results are plotted
    """
    # define array in which polynomial is fitted using the poly coefficients
    fit_vals = np.arange(-50, 100, 0.01)
    fit = poly.polyval(fit_vals, coefs)
    fit_range_begin, fit_range_end = use_vals[0], use_vals[-1]

    ew_fit_begin = (coefs[2] * fit_range_begin**2 + coefs[1] *
                    fit_range_begin + coefs[0])
    ew_fit_end = (coefs[2] * fit_range_end**2 + coefs[1] * fit_range_end +
                  coefs[0])

    meas_ew_min, meas_ew_max = meas_ew - meas_ewerr, meas_ew + meas_ewerr

    # convert the measured ew into a number corresponding to a spectral type
    specval, specval_u, specval_l = get_spectype(fit_vals, fit, meas_ew,
                                                 meas_ewerr)

    specval, specval_u, specval_l = specval[1], specval_u[1], specval_l[1]

    # if an axis is given: plot the polynomial as well as the measured SpT
    if ax != 'None':
        # Value within range of fitted polynomial (no extrapolation)
        if (((meas_ew_min) <= ew_fit_begin) & ((meas_ew_max) >= ew_fit_end)):
            specerr1 = abs(specval-specval_u)
            specerr2 = abs(specval-specval_l)

            spec_err = [[specerr1], [specerr2]]
            ax.errorbar(specval, meas_ew, xerr=spec_err, yerr=meas_ewerr,
                        fmt='o', color='mediumblue', label='MUSE',
                        markersize=7, mec='mediumblue')

            ax.add_patch(Rectangle((specval_u, -10), specerr1+specerr2,
                                   30, color='darkcyan', alpha=0.3))

        # EW larger than determined value => Left outside range.
        elif meas_ew_min > ew_fit_begin:
            ax.add_patch(Rectangle((-10, -10), use_vals[1], 30,
                                   color='darkcyan', alpha=0.3))
            ax.plot([-10, fit_range_begin], [meas_ew, meas_ew],
                    color='mediumblue', label='MUSE')
            ax.add_patch(Rectangle((-10, meas_ew_min), fit_range_begin+10,
                                   meas_ew_max, color='mediumblue', alpha=0.4))

        # Value smaller than determined value => Right outside range
        elif meas_ew_max < ew_fit_end:
            ax.add_patch(Rectangle((fit_range_end, -10), 30, 30,
                                   color='darkcyan', alpha=0.3))
            ax.plot([fit_range_end, 30], [meas_ew, meas_ew],
                    color='mediumblue', label='MUSE')
            ax.add_patch(Rectangle((fit_range_end, meas_ew_min), 30,
                                   2 * meas_ewerr, color='mediumblue',
                                   alpha=0.4))

    return(specval, specval_u, specval_l)


def get_spt(ew, ewerr, ax, lim=0.1, line_to_consider=6678):
    # get the HERMES calibration
    (vals, use_vals, hermes_ews, hermes_ewerrs, use_ews, use_ewerrs,
     coefs) = prep_herm(line=line_to_consider)

    # get the determined spectral values for the data and plot them
    (specval, specval_u, specval_l) = check_range(use_vals, coefs, ew, ewerr,
                                                  ax)

    ax.set_title(r' Spectral type from %s $\AA$' % line_to_consider)

    # plot the ew value from calibration of standard star spectra
    ax.errorbar(vals, hermes_ews, xerr=0, yerr=hermes_ewerrs, fmt='-o',
                color='dimgray', linestyle='None', mec='k', markersize=4,
                alpha=0.8, label='HERMES')

    # overplot the standard star values used for the fit of the polynomial
    ax.errorbar(use_vals, use_ews, xerr=0, yerr=use_ewerrs, markersize=8,
                linestyle='None', fmt='-x', alpha=0.8, color='k',
                label='used HERMES')

    # plot the interpolated polynomial
    fit_plot = poly.polyval(use_vals, coefs)
    ax.plot(use_vals, fit_plot, color='k', alpha=0.8)

    # plot a line around 0 and the excluded region around it
    ax.axhline(y=0.0, color='k', linestyle='-', alpha=0.8)
    ax.add_patch(Rectangle((-6, 0-lim), 30., 2*lim, color='darkgray',
                           alpha=0.3))

    # input values for new x ticks
    xtick_vals = np.arange(-2, 12, 1)
    labs = ['O7', 'O8', 'O9', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
            'B8', 'B9', 'A0']
    ax.set_xlim([-3, 12])
    ax.set_ylim([-0.1, 1.0])
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(labs)
    ax.grid(b=True, which='major', linestyle=':', alpha=0.7)

    ax.set_ylabel(r'Equivalent width [$\AA$]')
    ax.set_xlabel('Spectral type')
    ax.legend(loc=7)

    return specval, specval_l, specval_u


def convert_spectype(specval):
    # converts number (i.e. from plotting or table) into spectral type
    # numbers are set up in a way that B0 = 1
    specvals = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17]
    spectypes = ['O7', 'O8', 'O9', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                 'B7', 'B8', 'B9', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']

    if ((specval > 20) or (specval < 0)):
        spec_letter = '???'
    else:

        idx = specvals.index(round(specval))
        spec_letter = spectypes[idx]
    return spec_letter


def convert_spt_to_number(spectype):
    # converts SpT into number value
    # numbers are set up in a way that B0 = 1
    specvals = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17]
    spectypes = ['O7', 'O8', 'O9', 'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                 'B7', 'B8', 'B9', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']

    if spectype in spectypes:
        idx = spectypes.index(spectype)
        specval = specvals[idx]
    else:
        specval = np.nan
    return specval
