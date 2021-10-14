import matplotlib.pyplot as plt
import astro_utils.input_output as inout
from astropy.io import fits
import sys


fig, ax = plt.subplots()

for i in range(len(sys.argv)-1):
    j = i+1
    infile = sys.argv[j]

    wave_in, flux_in = inout.read_file(infile)

    ext = str(infile).split('/')[-1].split('.')[1]
    if ext == 'fits':
        header = fits.getheader(infile)
        if 'OBJECT' in header:
            targetname = header['OBJECT']
            if 'MJD-OBS' in header:
                MJD = header['MJD-OBS']
                name = targetname + ', MJD: ' + str(round(MJD, 2))
            elif 'BJD' in header:
                BJD = header['BJD']
                MJD = BJD - 2400000.5
                name = targetname + ', MJD: ' + str(round(MJD, 2))
            else:
                name = targetname
        else:
            name = str(infile).split('/')[-1].split('.')[0]

    else:
        name = str(infile).split('/')[-1].split('.')[0]

    if len(flux_in) == 2:
        flux = flux_in[0]
        err = flux_in[1]
    else:
        flux = flux_in
        err = None

    # Do the plotting
    ax.plot(wave_in, flux, linewidth=1.0, alpha=0.8, label=name)

#   if err is not None:
#       ax.fill_between(wave_in, flux-err, flux+err, color='k', alpha=0.1)
ax.legend(loc='upper right')
ax.set_xlabel("Wavelength [A]")
ax.set_ylabel("Flux")
plt.show()
