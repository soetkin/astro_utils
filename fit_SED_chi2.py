import json
import time
import numpy as np
from os.path import isfile
from astro_utils import input_output as inout
from astro_utils import constants as const
from astro_utils.sed_funcs import sed_functions as sed
from astro_utils.sed_funcs import sed_plots


##########################################################################
time_start = time.time()
##########################################################################

# Distance and extinction
dist = 116  # pc
E_BV = 0.4  # Mean E(B-V) value, Used with ccm89 and f19
R_V = 3.1	# Used with ccm89 and f19
A_V = 1	# Used for fm07 and ccm89

# The extinction model to follow (
#		implemented: 
#			- Fitzpatrick 2019 (f19) ; 0.115 - 3.3 micron (https://dust-extinction.readthedocs.io/en/stable/dust_extinction/choose_model.html)
#			- Fitzpatrick & Massa 2007 (fm07) ; 0.91 - 6 micron (https://extinction.readthedocs.io/en/latest/)
#			- Cardelli, Clayton & Mathis (ccm89) ; 1.25 - 3.3 micron (https://extinction.readthedocs.io/en/latest/api/extinction.ccm89.html#extinction.ccm89)
ext_model = 'fm07'

##########################################################################
# Model path
m_path = './models/BGflux_v2/'
# fit ranges
teff_range = np.arange(15000, 31000, 2000)
logg_range = np.arange(300, 500, 50)
radi_range = np.arange(3, 10, 0.1) * const.Rsun_pc

##########################################################################
# Filter information
f_path = './filter_funcs/'
f_curve_path = 'filter_funcs/filter_curves/'
with open(f_path + 'filters.json') as f:
    master_filter = json.load(f)

# filter names that should be used
fnames = ['GAIA3.Gbp', 'GAIA3.G', 'GAIA3.Grp', '2MASS.J', '2MASS.H',
          '2MASS.Ks', 'Hipparcos.Hp', 'TYCHO.B', 'TYCHO.V']  # , 'WISE.W1']  # , 'WISE.W2', 'WISE.W3', 'WISE.W4']

# corresponding filter names from vizier
fnames_vizier = ['GAIA/GAIA3:Gbp', 'GAIA/GAIA3:G', 'GAIA/GAIA3:Grp',
                 '2MASS:J', '2MASS:H', '2MASS:Ks', 'HIP:Hp', 'HIP:BT', 'HIP:VT']
# 'WISE:W1'] , 'WISE:W2', 'WISE:W3', 'WISE:W4']

# read in transmission curves of all filters to use
wcens_filter, waves_filter, fluxes_filter, labs_filter = [], [], [], []
for fname in fnames:
    fil = [f for f in master_filter if f['filtername'] == fname][0]
    fil_file = fil['instrument'] + '_' + fil['filtername'] + '.txt'
    wave_f, flux_f = np.loadtxt(f_curve_path + fil_file, unpack=True)
    wcens_filter.append(fil['lambda_eff'])
    labs_filter.append(fil['filtername'])
    waves_filter.append(wave_f)
    fluxes_filter.append(flux_f)


##########################################################################
# FITTING
##########################################################################
obs_infile = './22Sco.vot'
outfilename = './outfile_sed_22Sco_chi2_' + ext_model + '.csv'

# read in the observations
w_obs, f_obs, ferr_obs = sed.prep_obs_vot(obs_infile, fnames, fnames_vizier,
                                          plot_flag=True)

if isfile(outfilename):
    print("Specified output file exists already ... no fitting done")
else:
    print("Started the SED fit")
    sed.fit_sed(f_obs, ferr_obs, dist, E_BV, R_V, A_V,  # observations
                waves_filter, fluxes_filter,  # filters
                teff_range, logg_range, radi_range, m_path,  # models
                outfilename, ext_model = ext_model)

time_end = time.time()
print('Took ' + str(round(time_end-time_start)) + ' s to fit SED.')
print('---------------------------------------------------------------\n')
##########################################################################
# Plotting of the results
##########################################################################
print('Making the corner plot now ...')

# Make the corner plot, return best-fit teff, logg and radius
teff_val, logg_val, radius_val = sed_plots.plot_corner(outfilename,
                                                       'corner_22Sco_' + ext_model + '.pdf')

# now: compute flux in best-fitting model
# tlusty file name
print('Plotting the SED ...')
model = 'BG' + str(teff_val) + 'g' + str(round(logg_val*100)) + 'v2.flux.gz'

# read in model (eddington flux at the stellar surface)
wave_m, flux_m = inout.read_tlusty(m_path + model, sed = True)

# radius conversion into pc
radius_pc = radius_val * const.Rsun_pc
# prepare the SED, scale it for distance and radius, apply extinction
wave_c, f_red = sed.prep_sed(wave_m, flux_m, radius_pc, dist, E_BV, R_V, ext_model = ext_model, A_V = A_V)

# measure the flux of the best fit model in the respective filter bands
f_model, wave_conv_fils, flux_conv_fils = [], [], []
for f in range(len(waves_filter)):
    wave_f, flux_f = waves_filter[f], fluxes_filter[f]

    (wave_f, flux_f,
     int_flux_f) = sed.conv_filter(wave_c, f_red,  wave_f, flux_f)

    f_model.append(int_flux_f)
    wave_conv_fils.append(wave_f)
    flux_conv_fils.append(flux_f)

sed_plots.plot_sed(w_obs, f_obs, ferr_obs,  # observed fluxes
                   wcens_filter, f_model,  # fluxes measured in best-fit model
                   wave_conv_fils, flux_conv_fils,  # flux-convolved filters
                   labs_filter, 'best_fit_SED_22Sco_' + ext_model + '.pdf')  # filter names
print('===============================================================')
