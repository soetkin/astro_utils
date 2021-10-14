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
E_BV = 0.2  # Mean E(B-V) value
R_V = 3.1

##########################################################################
# Model path
m_path = './grid_fits/'
# fit ranges
teff_range = np.arange(15000, 31000, 2000)
logg_range = np.arange(300, 500, 50)
radi_range = np.arange(3, 10, 0.1) * const.Rsun_pc

##########################################################################
# Filter information
f_path = './filters/'
with open(f_path + 'filters.json') as f:
    master_filter = json.load(f)

# filter names that should be used
fnames = ['GAIA2r.Gbp', 'GAIA2r.G', 'GAIA2r.Grp', '2MASS.J', '2MASS.H',
          '2MASS.Ks']  # , 'WISE.W1']  # , 'WISE.W2', 'WISE.W3', 'WISE.W4']

# corresponding filter names from vizier
fnames_vizier = ['GAIA/GAIA2:Gbp', 'GAIA/GAIA2:G', 'GAIA/GAIA2:Grp',
                 '2MASS:J', '2MASS:H', '2MASS:Ks']
# 'WISE:W1'] , 'WISE:W2', 'WISE:W3', 'WISE:W4']

# read in transmission curves of all filters to use
wcens_filter, waves_filter, fluxes_filter, labs_filter = [], [], [], []
for fname in fnames:
    fil = [f for f in master_filter if f['filtername'] == fname][0]
    fil_file = fil['instrument'] + '_' + fil['filtername'] + '.txt'
    wave_f, flux_f = np.loadtxt(f_path + fil_file, unpack=True)
    wcens_filter.append(np.mean(wave_f))
    labs_filter.append(fil['filtername'])
    waves_filter.append(wave_f)
    fluxes_filter.append(flux_f)


##########################################################################
# FITTING
##########################################################################
obs_infile = './22Sco_test/22Sco.vot'
outfilename = './outfile_sed_22Sco_chi2.csv'

# read in the observations
w_obs, f_obs, ferr_obs = sed.prep_obs_vot(obs_infile, fnames, fnames_vizier,
                                          plot_flag=True)

if isfile(outfilename):
    print("Specified output file exists already ... no fitting done")
else:
    print("Started the SED fit")
    sed.fit_sed(f_obs, ferr_obs, dist, E_BV, R_V,  # observations
                waves_filter, fluxes_filter,  # filters
                teff_range, logg_range, radi_range, m_path,  # models
                outfilename)

time_end = time.time()
print('Took ' + str(round(time_end-time_start)) + ' s to fit SED.')
print('---------------------------------------------------------------\n')
##########################################################################
# Plotting of the results
##########################################################################
print('Making the corner plot now ...')

# Make the corner plot, return best-fit teff, logg and radius
teff_val, logg_val, radius_val = sed_plots.plot_corner(outfilename,
                                                       'corner_22Sco.pdf')

# now: compute flux in best-fitting model
# tlusty file name
print('Plotting the SED ...')
model = 'BG' + str(teff_val) + 'g' + str(round(logg_val*100)) + 'v2_sed.fits'

# read in model (eddington flux at the stellar surface)
wave_m, flux_m = inout.read_tlusty_fits(m_path + model, fluxtype='fluxspec')

# radius conversion into pc
radius_pc = radius_val * const.Rsun_pc
# prepare the SED, scale it for distance and radius, apply extinction
wave_c, f_red = sed.prep_sed(wave_m, flux_m, radius_pc, dist, E_BV, R_V,
                             w_min=waves_filter[0][0],
                             w_max=waves_filter[-1][-1])

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
                   wcens_filter, f_model,  # cluxes measured in best-fit model
                   wave_conv_fils, flux_conv_fils,  # flux-convolved filters
                   labs_filter, 'best_fit_SED_22Sco.pdf')  # filter names
print('===============================================================')
