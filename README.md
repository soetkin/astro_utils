ASTRO-UTILS

-- Utility package containing astronomy tools mainly for spectral analysis and SED fitting --

v0.1  14.10.2021

Author: Julia Bodensteiner

For questions, please contact:  julia.bodensteiner@eso.org



CONTENTS
-------------------------------------------------------------------------------------------------------------------
1. input_output.py
   - reading of different types of files containing spectra, including 
     + two-column txt
     + several fits, including HERMES, ESO (FEROS, UVES, MUSE, ...), SALT, ... files
     + atmosphere models from TLUSTY / GSSP
   - writing of spectra 
     + fits with corresponding headers
     + 2-column ascii
     + also: 2D images
   - functions to cut spectral waverange and mark MUSE laser region

2. constants.py
   - often-used constants

3. plot_funcs: 
 3.1 mark_lines.py
   - utility package to mark spectral lines in a plotted spectrum

 3.2 plot.py
   - simple plotting script to plot a 1D spectrum

4. spec_funcs:
 4.1 normalization_epsf_cleaned.py
   - automatic normalization script
   - written for low-resolution MUSE spectra
   - continuum selection by comparison of median- and min-max-filter
   - input:  (flux-calibrated) spectrum
   - output: normalized spectrum

 4.2 spectral_typing.py
   - automatic spectral typing of MUSE spectra using standard stars observed with HERMES

 4.3 spec_functions.py

5. extrac_funcs
 5.1 photometry_functions.py

6. sed_funcs
 6.1 sed_functions.py
 6.2 sed_plots.py

7. fit_funcs
 7.1 fit_functions.py
