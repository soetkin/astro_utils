import json
import requests
import numpy as np


"""
Script to prepare additional filters for later usage. 
All filters are summarized in ./filters/filters.json
For non-exisiting filters, the transmission curve is downloaded into ./filters
Filter data is uniquely taken from svo2 webpage (see below)
Their naming convention is followed.
Requires: Change of USER INPUT
  - path to filter directory
  - name of instrument and filter that is required
  - effective wavelength of the filter; unit: Angstrom
  - flux zeropoint in Vega system; unit: erg/s cm^2 A
"""

url_prefix = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='

############################## USER INPUT ################################
# location of the master filter file and all transmission curves
filter_path = './'


# information from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php #

instrument = 'Hipparcos'
filter_name = 'Hipparcos.Hp'
lambda_eff = 4897.85  # Angstrom, mathematical properties lambda_eff
flux_zeropoint = 4.39E-9  # erg/s cm^2 A, Zero point in Vega system
##########################################################################

# check if the filter exists already in json filter file
filter_infile = filter_path + 'filters.json'
with open(filter_infile) as f:
    master_filter = json.load(f)

# check if filter already exists
exists = False
for f in master_filter:
    if f["instrument"] == instrument and f["filtername"] == filter_name:
        exists = True

if exists is True:
    print("Filter %s already exists in %s" % (filter_name, filter_infile))

else:
    print("Prepping filter %s ..." % filter_name)
    # add filter to json file
    # make information a dictionary
    new_filter = {'instrument': instrument,
                  'filtername': filter_name,
                  'lambda_eff': lambda_eff,
                  'fzeropoint': flux_zeropoint}

    # append new filter to master_filter dictionary
    master_filter.append(new_filter)
    with open(filter_infile, 'w') as json_file:
        json.dump(master_filter, json_file, indent=3)
    
    
    # create the filter transmission curve file
    url = url_prefix + instrument + '/' + filter_name
    
    resp = requests.get(url)
    
    outfname = filter_path + instrument + '_' + filter_name + '.txt'
    
    with open(outfname, 'wb') as outfile:
        outfile.write(resp.content)