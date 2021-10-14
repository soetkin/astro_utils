import json
import numpy as np
import matplotlib.pyplot as plt

filter_path = './filters/'
filter_infile = filter_path + 'filters.json'
with open(filter_infile) as f:
    master_filter = json.load(f)


# print all available filters:
print("Currently available filters are ... ")
for f in master_filter:
    print(f["filtername"])

# select a filter and print the important information
fname = '2MASS.J'
fil = [filter for filter in master_filter if filter['filtername'] == fname][0]
print(fil)


# prepare a plot with several filters
fnames = ['WISE.W1', 'WISE.W2', 'WISE.W3', 'WISE.W4']
fig, ax = plt.subplots()

for fname in fnames:
    fil = [f for f in master_filter if f['filtername'] == fname][0]
    fil_file = (filter_path + fil['instrument'] + '_' + fil['filtername'] +
                '.txt')
    wave, flux = np.loadtxt(fil_file, unpack=True)

    ax.plot(wave, flux, alpha=0.7, label=fil['filtername'])

ax.legend()
ax.set_ylabel('Transmission')
ax.set_xlabel(r'Wavelength [$\AA$]')
plt.show()
