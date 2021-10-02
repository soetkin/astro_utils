ASTRO-UTILS

-- Utility package containing astronomy function mainly for spectra --

v0.1  02.10.2021

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

2. mark_lines.py
   - utility package to mark spectral lines in a plotted spectrum

3. plot.py
   - simple plotting script to plot a 1D spectrum

4. normalization_epsf_cleaned.py
   - automatic normalization script
   - written for low-resolution MUSE spectra
   - continuum selection by comparison of median- and min-max-filter
   - input:  (flux-calibrated) spectrum
   - output: normalized spectrum

5. spectral_typing.py

6. spec_functions.py

7. photometry_functions.py

8. fit_functions.py
