import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def max_filter(X, window):
    # Sliding window min max filter, X is input array (typically flux)
    max_array = []  # make empty array for maximum values
    min_array = []  # make empty array for minimum values
    for i in range(len(X)):  # go through input array from beginning to end
        Istart = max([0, i-window/2])  # define start point of each window
        Iend = min([len(X)-1, i+window/2])  # and end point of sliding window

        # find the maximum of the input array in this region
        max_val = np.nanmax(X[int(Istart):int(Iend)])
        max_array.append(max_val)  # append it to the output array
        min_val = np.nanmin(X[int(Istart):int(Iend)])  # do the same for mins
        min_array.append(min_val)

    return max_array, min_array


def med_filter(X, window):
    # Sliding window median filter, X is input array (typically flux)
    med_array = []  # make empty flux array for median values
    for i in range(len(X)):  # go through input array from beginning to end
        Istart = max([0, i-window/2])  # define start
        Iend = min([len(X)-1, i+window/2])  # and end point of sliding window

        # find the median of the input array in this region
        med_val = np.nanmedian(X[int(Istart):int(Iend)])
        med_array.append(med_val)

    return med_array


def first_clipping(wave, flux, err, window, param):
    # remove spectral lines by comparison of median and max/min max_filter
    # important: actual flux value is kept!
    med_array = med_filter(flux, window)
    max_array, min_array = max_filter(flux, window)

    ratio_max = np.array(max_array) / np.array(med_array)
    ratio_min = np.array(min_array) / np.array(med_array)

    # compare whether the ratio - 1. is smaller/bigger than param
    wave_cont, flux_cont, err_cont = [], [], []

    for i in range(len(med_array)):
        # select everything in continuum
        if ratio_max[i] - 1. < param and 1. - ratio_min[i] < 2 * param:
            wave_cont.append(wave[i])
            flux_cont.append(flux[i])
            err_cont.append(err[i])

    print("%i continuum points are selected in 1st clipping." % len(wave_cont))
    return wave_cont, flux_cont, err_cont


def compare_to_poly_pixel(wave, flux, err, polyflux, param):
    polyflux = np.array(polyflux)

    wave_cont, flux_cont, err_cont = [], [], []
    for i in range(len(flux)):
        # select everything in continuum
        if abs(flux[i]-polyflux[i]) < param:
            wave_cont.append(wave[i])
            flux_cont.append(flux[i])
            err_cont.append(err[i])
    return wave_cont, flux_cont, err_cont


def fit_poly(wave_in, wave_target, flux_in, deg):
    wave_in = np.array(wave_in)
    flux_in = np.array(flux_in)
    wave_fit = wave_in[np.logical_not(np.isnan(flux_in))]
    flux_fit = flux_in[np.logical_not(np.isnan(flux_in))]

    coefs = poly.polyfit(wave_fit, flux_fit, deg)
    fit = poly.polyval(wave_target, coefs)
    return fit


def mask_sky(wave, flux, err, wls_start, wls_end):
    # problem: hardcoded wave array but not sure how to fix it
    wave_A = np.arange(6017.30126953, 9349.80126953, 1.25)
    # mask bad sky subtraction
    wls_new_start, wls_new_end = [], []
    # wl for sky defined in A, convert to vals between 0 and 1 ( input array)
    for i in range(len(wls_start)):
        val_start, val_end = wls_start[i], wls_end[i]

        # get index of closest entry in Angstrom array
        index_start = np.abs(wave_A - val_start).argmin()
        new_start = wave[index_start]

        index_end = np.abs(wave_A - val_end).argmin()
        new_end = wave[index_end]

        wls_new_start.append(new_start)
        wls_new_end.append(new_end)

    for wl in range(len(wls_new_start)):
        wl_start, wl_end = wls_new_start[wl], wls_new_end[wl]
        ind = [j for j in range(len(wave)) if wl_start < wave[j] < wl_end]
        flux[ind] = np.nan
        err[ind] = np.nan

    return flux, err, wls_new_start, wls_new_end


def get_paschen(wave, flux, err):
    wave_A = np.arange(6017.30126953, 9349.80126953, 1.25)
    # wavelength points between Paschen lines where continuum is
    wave_paschen = [8488., 8527., 8575., 8635., 8710, 8805., 8943., 9060.,
                    9125., 9186., 9286., 9330.]  # A
    # wavelength at which contamination by Paschen series starts
    cut_wave_A = 8380.  # A
    # get index of closest entry in Angstrom array
    index_cut = np.abs(wave_A - cut_wave_A).argmin()
    # pick corresponding value in new wavelength array
    cut_wave = wave[index_cut]

    force_wave, force_flux, force_err = [], [], []
    wind = 10.
    # wl for sky defined in A, convert to vals between 0 and 1 ( input array)
    for i in range(len(wave_paschen)):
        # cut window around wave points in which average flux is computed
        val_start, val_end = wave_paschen[i] - wind, wave_paschen[i] + wind

        # get index of closest entry in Angstrom array
        index_start = np.abs(wave_A - val_start).argmin()
        new_start = wave[index_start]
        index_end = np.abs(wave_A - val_end).argmin()
        new_end = wave[index_end]

        # wave array in this window
        wave_arr = wave[(wave > new_start) & (wave < new_end)]
        flux_arr = flux[(wave > new_start) & (wave < new_end)]
        err_arr = err[(wave > new_start) & (wave < new_end)]

        # append the median flux value
        force_wave.append(np.nanmedian(wave_arr))
        force_flux.append(np.nanmedian(flux_arr))
        force_err.append(np.nanmedian(err_arr))

    return force_wave, force_flux, force_err, cut_wave


def norm_indi(wave, flux, err, wind, val, flag, snr=150., plot='no',
              star_id='None', return_cont='no'):
    """ normalization function:
        wave, flux, err: input arrays for wavelength, flux and errors
        wind: width of sliding window used for first clipping [A]
        val:  tolerance value for first clipping, tolerance = val / snr
        snr:   estimated snr in spectrum, if not given: set to 150.
        plot:  flag to save plots (default is no). If not no than give the
               filename that should be used.
        star_id:   ID of the star, used for plotting
    """
    poly_degree = 21
    wave_cp, flux_cp, err_cp = wave.copy(), flux.copy(), err.copy()

    ##########################################################################
    #  first clipping
    #  remove bad sky subtraction and wide spectral lines
    ##########################################################################

    # mask regions of the sky that are contaminated by the sky
    if flag == 'blue':
        print("Blue part of the spectrum")
        wave_mask, flux_mask, err_mask = wave_cp, flux_cp, err_cp
    elif flag == 'red':
        print("Red part of the spectrum")
        wls_sky_s_A = [6860., 7150., 7570., 8200.]  # Angstrom
        wls_sky_e_A = [6930., 7340., 7700., 8280.]  # Angstrom
        flux_sky, err_sky, wls_sky_s, wls_sky_e = mask_sky(wave_cp, flux_cp,
                                                           err_cp, wls_sky_s_A,
                                                           wls_sky_e_A)

        # get continuum wave points and median flux values in Paschen series
        wave_pas, flux_pas, err_pas, cut_wave = get_paschen(wave_cp, flux_sky,
                                                            err_sky)
        # cut off the wavelength array before the Paschen series
        wave_mask = wave_cp[wave_cp < cut_wave]
        flux_mask = flux_sky[wave_cp < cut_wave]
        err_mask = err_sky[wave_cp < cut_wave]

    # first clipping
    clip_param = val / snr
    wave_fc, flux_fc, err_fc = first_clipping(wave_mask, flux_mask, err_mask,
                                              wind, clip_param)

    # fit a polynomial through the clipped flux array
    polyfit = fit_poly(wave_fc, wave_fc, flux_fc, poly_degree)

    #########################################################################
    # fig, axes = plt.subplots(3, figsize=(12, 15), sharex=True)
    ########################################################################
    # ax1 = axes[0]
    # ax1.set_title('Star ID ' + str(star_id))
    # ax1.plot(wave, flux, marker='+', color='dimgray', label='input')
    # ax1.plot(wave_fc, flux_fc, marker='+', label='first clip')
    # ax1.plot(wave_fc, polyfit, linewidth=2, label='polyfit')

    # if flag == 'red':
    #     for wl in range(len(wls_sky_s)):
    #         wl_start, wl_end = wls_sky_s[wl], wls_sky_e[wl]
    #         wdt = wl_end - wl_start
    #         ax1.add_patch(Rectangle((wl_start, -1), width=wdt, height=10.,
    #                                 color='red', alpha=0.3))
    # ax1.legend()
    # ax1.set_ylabel("Flux")
    # ax1.grid(linestyle=':', alpha=0.6)
    #########################################################################

    ##########################################################################
    #  cut points around polynomial, do it iteratively
    ##########################################################################
    num_iter = 4
    cols = ['darkorange', 'forestgreen', 'midnightblue', 'crimson']

    cut_val = 0.5
    cut_param = cut_val / snr

    # ax2 = axes[1]
    # ax2.plot(wave, flux, zorder=0, marker='+', color='dimgray', label='input')
    # ax2.plot(wave_fc, polyfit, label='input polynomial')
    for it in range(num_iter):

        wave_cont, flux_cont, err_cont = compare_to_poly_pixel(
            wave_fc, flux_fc, err_fc, polyfit, cut_param)

        polyfit = fit_poly(wave_cont, wave_fc, flux_cont, poly_degree)

        # lab = 'polyfit thres: ' + "{:1.4f}".format(cut_param)
        # ax2.plot(wave_fc, polyfit, label=lab, color=cols[it], alpha=0.7)
        # ax2.plot(wave_cont, flux_cont, color=cols[it], alpha=0.7)
        cut_param = 0.6 * cut_param

    wave_cont, flux_cont, err_cont = compare_to_poly_pixel(
        wave_fc, flux_fc, err_fc, polyfit, cut_val / snr)

    if flag == 'red':
        # concatenate again with Paschen array
        wave_cont = np.concatenate((wave_cont, wave_pas))
        flux_cont = np.concatenate((flux_cont, flux_pas))

    polyfit_final = fit_poly(wave_cont, wave, flux_cont, poly_degree)

    # ax2.plot(wave_cont, flux_cont, zorder=0, marker='+', color='crimson',
    #          alpha=0.7)
    # ax2.plot(wave, polyfit_final, color='k', label='final')
    #
    # ax2.set_ylabel("Flux")
    # ax2.legend()
    # ax2.grid(linestyle=':', alpha=0.6)
    ##########################################################################

    ##########################################################################
    # normalize the spectrum
    ##########################################################################
    norm_flux = flux / polyfit_final
    norm_err = err / polyfit_final

    if flag == 'hello':
        fig, ax = plt.subplots(figsize=(12, 6))

        # fit polynomial through normalised spectrum
        flux_cont_norm, err_cont_norm = [], []
        for w in range(len(wave)):
            wave_indi = wave[w]
            if wave_indi in wave_cont:
                flux_cont_norm.append(norm_flux[w])
                err_cont_norm.append(norm_err[w])

        polyfit_norm = fit_poly(wave_cont, wave, flux_cont_norm, 21)

        ax.plot(wave, norm_flux, zorder=0, marker='+', color='dimgray',
                label='in')
        ax.plot(wave_cont, flux_cont_norm, color='crimson', marker='+',
                label='co')
        ax.plot(wave, polyfit_norm, color='k', linewidth=4, label='poly')
        ax.legend()

        # ax.axhline(y=1.0, color='k')
        ax.axhline(y=1.025, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=0.975, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1.01, color='k', linestyle=':', alpha=0.2)
        ax.axhline(y=0.99, color='k', linestyle=':', alpha=0.2)
        ax.set_ylim([0.80, 1.20])
        ax.set_ylabel("Normalized flux")
        ax.grid(linestyle=':', alpha=0.6)
        ax.set_xlim([min(wave)-0.01, max(wave)+0.01])
        #######################################################################

        #######################################################################
        # devide the already normalized spectrum by polynomial fitted to that
        #######################################################################
        norm_flux = norm_flux / polyfit_norm
        norm_err = norm_err / polyfit_norm

    # ax3 = axes[2]
    # ax3.plot(wave, norm_flux, marker='+', color='dimgray', label='in')
    # ax3.axhline(y=1.02, color='k', linestyle='--', alpha=0.3)
    # ax3.axhline(y=0.98, color='k', linestyle='--', alpha=0.3)
    # ax3.axhline(y=1.01, color='k', linestyle=':', alpha=0.2)
    # ax3.axhline(y=0.99, color='k', linestyle=':', alpha=0.2)
    # ax3.set_ylim([0.80, 1.20])
    # ax3.set_ylabel("Normalized flux")
    # ax3.grid(linestyle=':', alpha=0.6)
    # ax3.set_xlim([min(wave)-0.01, max(wave)+0.01])
    # plt.show()

    # if requested: save the plots as pdf file
    if plot != 'no':
        fig, axes = plt.subplots(3, figsize=(8.27, 11.00), dpi=300)  # A4 size
        step = 1.25
        ax1 = axes[0]
        if flag == 'blue':
            wave_n = np.arange(0, len(wave)) * step + 4599.865234375
            ax1.set_title('Star ID ' + str(star_id) + ' - blue part')

        elif flag == 'red':
            wave_n = np.arange(0, len(wave)) * step + 6017.30126953
            ax1.set_title('Star ID ' + str(star_id) + ' - red part')

        ax1.plot(wave, flux, marker='+', color='dimgray', label='input')
        ax1.plot(wave_fc, flux_fc, marker='+', label='first clip')
        ax1.plot(wave_fc, polyfit, linewidth=2, label='polyfit')

        if flag == 'red':
            for wl in range(len(wls_sky_s)):
                wl_start, wl_end = wls_sky_s[wl], wls_sky_e[wl]
                wdt = wl_end - wl_start
                ax1.add_patch(Rectangle((wl_start, -1), width=wdt, height=10.,
                                        color='red', alpha=0.3))
        ax1.legend()
        ax1.set_ylabel("Flux")
        ax1.grid(linestyle=':', alpha=0.6)

        cols = ['darkorange', 'forestgreen', 'midnightblue', 'crimson']
        ax2 = axes[1]
        ax2.plot(wave, flux, zorder=0, marker='+', color='dimgray',
                 label='input')
        ax2.plot(wave_fc, polyfit, label='input polynomial')

        wave_cont, flux_cont, err_cont = compare_to_poly_pixel(
            wave_fc, flux_fc, err_fc, polyfit, cut_val / snr)

        if flag == 'red':
            # concatenate again with Paschen array
            wave_cont = np.concatenate((wave_cont, wave_pas))
            flux_cont = np.concatenate((flux_cont, flux_pas))

        polyfit_final = fit_poly(wave_cont, wave, flux_cont, poly_degree)

        ax2.plot(wave_cont, flux_cont, zorder=0, marker='+', color='crimson',
                 alpha=0.7)
        ax2.plot(wave, polyfit_final, color='k', label='final')

        ax2.set_ylabel("Flux")
        ax2.legend()
        ax2.grid(linestyle=':', alpha=0.6)

        ax3 = axes[2]
        ax3.plot(wave_n, norm_flux, marker='+', color='dimgray', label='in')
        ax3.axhline(y=1.02, color='k', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.98, color='k', linestyle='--', alpha=0.3)
        ax3.axhline(y=1.01, color='k', linestyle=':', alpha=0.2)
        ax3.axhline(y=0.99, color='k', linestyle=':', alpha=0.2)
        ax3.set_ylim([0.80, 1.20])
        ax3.set_ylabel("Normalized flux")
        ax3.grid(linestyle=':', alpha=0.6)
        ax3.set_xlim([min(wave_n)-50, max(wave_n)+50])
        ax3.set_xlabel(r'Wavelength [$\AA$]')

        fig.savefig(plot, bboxinches='tight')
        plt.close()
    if return_cont == 'no':
        return wave, norm_flux, norm_err
    elif return_cont == 'yes':
        return wave, polyfit_final
