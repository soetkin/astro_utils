import pandas as pd
import matplotlib.pyplot as plt
from astro_utils.fit_funcs import fit_functions as fit
from scipy.stats.distributions import chi2 as scipy_chi2


def plot_corner(tab_file, cornerfilename='corner.pdf'):
    """
    create corner plot with results from SED fit
    get best-fit model as the one with the lowest chi^2
    compute confidence leven (95% significance)
    compute errors from confidence level
    input: output file from SED fitting with teff, logg, radius, dof, chi^2
           per model (for now: each parameter needs at least three values for
           interpolation to work), name of plot file
    returns: corner plot saved in specified filename
             best-fit values for teff, logg, radius
    """
    tab = pd.read_csv(tab_file)

    # read in unscaled chi2
    unscaled_chi2 = tab['chi2']
    # number of degrees of freedom
    dof = tab['dof'][0]

    # renormalize the chi2 such that the best fit corresponds to a chi2 = dof
    # which is similar to setting the reduced chi2 =1
    chi2 = unscaled_chi2 / unscaled_chi2.min() * dof

    # get confidence level
    dof = tab['dof'][0]
    conf_level = scipy_chi2.ppf(0.95, dof)

    # read in teffs, loggs and radius from SED output file
    teff, logg, radius = tab['teff'], tab['logg'], tab['radius']

    # interpolate between the minima
    (teff_i, teff_chi,
     teff_arr, teff_interp) = fit.interp_models(teff, chi2)
    (logg_i, logg_chi,
     logg_arr, logg_interp) = fit.interp_models(logg, chi2)
    (radius_i, radius_chi, radius_arr,
     radius_interp) = fit.interp_models(radius, chi2)

    # get minimum of chi2
    idx_min = chi2.idxmin()

    # get best-fit best-fit teff, logg and radius and respective errors
    teff_val = teff[idx_min]
    teff_l, teff_u = fit.get_errs(teff_arr, teff_interp, conf_level, teff_val)

    teff_val = round(teff[idx_min], -2)
    teff_l, teff_u = fit.get_errs(teff_arr, teff_interp, conf_level, teff_val)
    teff_l, teff_u = round(teff_l, -2), round(teff_u, -2)

    logg_val = logg[idx_min]
    logg_l, logg_u = fit.get_errs(logg_arr, logg_interp, conf_level, logg_val)
    logg_l, logg_u = abs(logg_l), abs(logg_u)

    radius_val = radius[idx_min]
    radius_l, radius_u = fit.get_errs(radius_arr, radius_interp, conf_level,
                                      radius_val)

    minima = [teff_val, logg_val, radius_val]

    # COOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORNER PLOT
    pars = [teff, logg, radius]

    # might have to be adjusted for corner plot to look nice
    vmin, vmax = min(chi2), min(chi2) + 0.003*(max(chi2)-min(chi2))

    # general plot settings
    fig = plt.figure(figsize=(7, 8))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.83, top=0.97,
                        wspace=0.05, hspace=0.05)

    colnames = 'teff', 'logg', 'radius'
    step_teff, step_logg = 1000, 0.25
    step_radius = 0.1

    # x limits and x labels
    xlims = [[min(teff)-2*step_teff, max(teff)+2*step_teff],
             [min(logg)-2*step_logg, max(logg)+2*step_logg],
             [min(radius)-2*step_radius, max(radius)+2*step_radius]]
    xlabels = ['Teff [K]', 'logg [dex]', 'radius [Rsun]']

    # make all different corner plot boxes
    for jj in range(len(pars)):
        for ii in range(len(pars)):
            ai = jj * (len(pars)) + ii + 1

            # diagonal
            if ii == jj:
                ax = fig.add_subplot(len(pars), len(pars), ai)

                # configure ticks, limits and labels
                if jj == 0:
                    ax.set_ylabel(r'$\chi^2$', fontsize=12)
                    ax.set_xticks([])
                    ax.set_xlim(xlims[ii])
                    ax.set_ylim(vmin, vmax)
                if ii == 1:
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_xlim(xlims[ii])
                    ax.set_ylim(vmin, vmax)
                if ii == len(pars)-1:
                    ax.set_xlabel(xlabels[ii], fontsize=12)
                    ax.set_xlim(xlims[ii])
                    ax.set_yticks([])
                    ax.set_ylim(vmin, vmax)

                # actual plotting
                ax.plot(pars[ii], chi2, 'k.', alpha=0.5)
                ax.axhline(conf_level, color='crimson', lw=1, alpha=0.7)
            elif ii < jj:
                ax = fig.add_subplot(len(pars), len(pars), ai)

                if ii == 0 and jj == 1:
                    xx = 2
                elif ii == 0 and jj == 2:
                    xx = 1
                elif ii == 1 and jj == 2:
                    xx = 0

                mask = (tab[colnames[xx]] == minima[xx])
                mtab = tab[mask]
                cscale = mtab['chi2'] / unscaled_chi2.min() * dof

                cb = ax.scatter(mtab[colnames[ii]], mtab[colnames[jj]],
                                c=cscale, marker='s', alpha=0.9,
                                cmap='inferno_r', vmin=vmin, vmax=vmax)
                ax.set_ylim(xlims[jj])
                ax.set_xlim(xlims[ii])

                # configure ticks, limits and labels
                if jj == len(pars)-1:
                    ax.set_xlabel(xlabels[ii], fontsize=12)
                    if ii != 0:
                        ax.set_yticks([])
                    else:
                        ax.set_ylabel(xlabels[jj], fontsize=12)
                elif ii == 0:
                    ax.set_ylabel(xlabels[jj], fontsize=12)
                    if jj != (len(pars)-1):
                        ax.set_xticks([])
                else:
                    ax.set_yticks([])

    # color bar configuration
    cbar_ax = fig.add_axes([0.83, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cb, cax=cbar_ax)
    cbar.set_label(r'$\chi^2$', fontsize=12)

    # print the best fit paramters
    print('Your best fit is:')
    print(' Teff = {:.0f}'.format(teff_val) + ' +{:.0f}'.format(teff_u) +
          ' -{:.0f}'.format(teff_l) + ' K')
    print(' logg = {:.1f}'.format(logg_val) + ' +{:.1f}'.format(logg_u) +
          ' -{:.1f}'.format(logg_l) + ' dex')
    print(' radius = {:.1f}'.format(radius_val) + ' +{:.1f}'.format(radius_u) +
          ' -{:.1f}'.format(radius_l) + ' R_sun')

    # write best fit parameters into plot
    ax_text = fig.add_subplot(3, 3, 3)

    ax_text.text(-0.1, 0.65, r' Teff = ${:.0f}'.format(teff_val) +
                 '^{{+{:.0f}}}'.format(teff_u) +
                 '_{{-{:.0f}}}$'.format(teff_l) + ' K')
    ax_text.text(-0.1, 0.5, r' logg = ${:.1f}'.format(logg_val) +
                 '^{{+{:.1f}}}'.format(logg_u) +
                 '_{{-{:.1f}}}$'.format(logg_l) + ' dex')
    ax_text.text(-0.1, 0.35, r' radius = ${:.1f}'.format(radius_val) +
                 '^{{+{:.1f}}}'.format(radius_u) +
                 '_{{-{:.1f}}}$'.format(radius_l) + r' R$_{\odot}$')

    ax_text.set_axis_off()
    fig.savefig(cornerfilename, bbox_inches='tight')
    plt.show()

    return teff_val, logg_val, radius_val  # [radius] = Rsun


def plot_sed(w_obs, f_obs, ferr_obs,  # observed fluxes
             w_mod, f_mod,  # flux measured for best fitting model
             wave_conv_fil, flux_conv_fil,  # filter transmission curves
             labs, sedfilename='best_fit_sed.pdf'):  # filter and plotfile name
    """
    plot observed fluxes, fluxes computed from best-fit model in all filters,
    and filters*model
    input: observations, fluxes for best-fit model, convolved filters, filter
           names, name of plot file
    returns: sed plot
    """
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for f in range(len(w_mod)):
        ax1.scatter(w_mod[f], f_mod[f], alpha=0.7)
        ax1.plot(wave_conv_fil[f], flux_conv_fil[f], alpha=0.7, label=labs[f])

    ax1.errorbar(w_obs, f_obs, yerr=ferr_obs,
                 ls='None', color='k', label='observations', marker='+')

    ax1.set_yscale('log')
    ax1.set_xlabel(r'Wavelength [$\AA$]')
    ax1.set_ylabel(r'Flux [erg/cm$^2$/s/$\AA$]')
    ax1.legend(loc='lower right')

    fig1.savefig(sedfilename, bbox_inches='tight')
    plt.show()
