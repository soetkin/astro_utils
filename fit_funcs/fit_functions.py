from scipy.stats.distributions import chi2 as scipy_chi2
import scipy.interpolate as scInterp
import numpy as np


def interp_models(parameter, chi2_array):
    # make array with the minima of the parameter (unique values)
    param_u = np.unique(parameter)
    param_chi = []
    # if the parameter was not explored:

    if len(param_u) < 3:
        print("Too few values to interpolate.")
        param_chi, param_arr, param_interp = np.nan, np.nan, np.nan
    # interpolate between the minimum chi2 of the different models
    else:
        # get chi2 for the minima
        for i in range(len(param_u)):
            param_chi.append(np.min(chi2_array[parameter == param_u[i]]))

        # interpolate chi2 minima
        interp = scInterp.interp1d(param_u, param_chi, kind='cubic',
                                   fill_value="extrapolate")
        # make finer grid of parameter to interpolate minima on
        min, max = np.min(param_u), np.max(param_u)  # + 0.01*np.min(param_u)
        step = (param_u[1]-param_u[0]) / 10
        param_arr = np.arange(min, max, step)

        # map minima to new array
        param_interp = interp(param_arr)
    return param_u, param_chi, param_arr, param_interp


def get_conf_level(chi2_red, dof, p_thres=0.05):
    # scaling factor for chi2 => renormalize so that min(reduced_chi2) = 1
    scaled_reduced_chi2 = chi2_red / np.nanmin(chi2_red)

    # input for gamma function: chi2, not reduced chi2
    scaled_chi2 = scaled_reduced_chi2 * dof

    # p-values from probability distribution of chi2
    # all models with p > xxx % are statistically equivalent
    p_val = scipy_chi2.sf(scaled_chi2, dof)

    # interpolate p-values to find chi2 value corresponding to p-value = xxx %
    new_chi2_arr = np.arange(np.nanmin(chi2_red), np.nanmax(chi2_red), 0.01)
    interp = scInterp.interp1d(chi2_red, p_val)
    p_val_interp = interp(new_chi2_arr)

    # find value closest to threshold p-val interpolated array and corresp chi2
    idx_chi2 = np.abs(p_val_interp - p_thres).argmin()
    conf_level = new_chi2_arr[idx_chi2]

    return conf_level


def compute_chi2_red(obs, model, obs_err, n_free_params):
    chi2 = 0
    for i in range(len(obs)):
        # if only one error is given for the entire range
        if type(obs_err) is float:
            chi2 += (obs[i] - model[i])**2 / (obs_err)**2
        # if each observation has it's own error
        else:
            chi2 += (obs[i] - model[i])**2 / (obs_err[i])**2

    # dof = len(obs) - n_free_params
    dof = len(obs) - n_free_params

    # if len(data points) = len(free parameters) => dof = 0 => set to 1
    if dof == 0:
        dof = 1

    red_chi2 = chi2 / dof
    return red_chi2, dof


def compute_chi2(obs, model, obs_err):

    chi2 = 0
    for i in range(len(obs)):
        # if only one error is given for the entire range
        if type(obs_err) is float:
            chi2 += (obs[i] - model[i])**2 / (obs_err)**2
        # if each observation has it's own error
        else:
            chi2 += (obs[i] - model[i])**2 / (obs_err[i])**2

    return chi2


def get_errs(param_arr, param_interp, conf_level, val_min):
    # Schnittpunkt zwischen zwei Kurven
    idxs = np.argwhere(np.diff(np.sign(param_interp - conf_level))).flatten()
    if len(idxs) == 0:
        print("No errors could be determined.")
        err_l = np.nan
        err_u = np.nan

    elif len(idxs) == 1:
        print("Only one error determined")
        if param_arr[idxs] < val_min:
            err_l = val_min - param_arr[idxs][0]
            err_u = np.nan
        elif param_arr[idxs] > val_min:
            err_u = param_arr[idxs][0] - val_min
            err_l = np.nan
    elif len(idxs) == 2:
        err_l = abs(val_min - param_arr[idxs[0]])
        err_u = abs(param_arr[idxs[1]] - val_min)

    elif len(idxs) > 2:
        # more that one interception with conf level => lowest and highest
        err_l = abs(val_min - param_arr[idxs[0]])
        err_u = abs(param_arr[idxs[-1]] - val_min)

    return(err_l, err_u)
