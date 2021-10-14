import json
import requests
import numpy as np
from astropy.io.votable import parse
import sys

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

#url_prefix = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='
url_prefix = 'http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?'
url_curve = 'http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id='

############################## USER INPUT ################################
# location of the master filter file and all transmission curves
filter_path = './'
filter_curves_path = './filter_curves/'

# information from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php #

#Searches by facility: returns all 2MASS filters in this instance.
# facility = 'Hipparcos'
facilities = sys.argv[1:]
##########################################################################

# check if the filter exists already in json filter file
filter_infile = filter_path + 'filters.json'
with open(filter_infile) as f:
    master_filter = json.load(f)

for i, fac in enumerate(facilities):	

	url_fac = url_prefix + 'Facility=' + fac

	resp = requests.get(url_fac)

	outfname = filter_path + fac + '.xml'

	with open(outfname, 'wb') as outfile:
	    outfile.write(resp.content)

	votable = parse(outfname)

	for table in votable.iter_tables():
		data = table.array
		print()
		print('Instrument: ' + fac)
		print('These are all available filters for the instrument.') 
		print(np.array(data['filterID']))
		print('Do you want to select them all?')
		select = input('[y/n] ')
		print()
		if select == 'n':
			print('Please type the indices of the filters you want with comma separation, starting from 0:')
			filters_select = np.array(input('').split(','), int)
			print()
		elif select == 'y':
			filters_select = np.arange(0, len(np.array(data['filterID']))+1, 1)
		else:
			print('no filters chosen, exiting...')
			exit()
		for n,dat in enumerate(data):
			if n not in filters_select:
				continue
			filter_id = data['filterID'][n]
			filter_id = filter_id.decode("utf-8").split('/')[-1]
			# check if filter already exists
			exists = False
			for f in master_filter:
				if f["instrument"] == fac and f["filtername"] == filter_id:
					exists = True

				if exists is True:
					print("Filter %s already exists in %s" % (filter_id, filter_infile))
					break
			else:
				print("Prepping filter %s ..." % filter_id)
				lambda_eff = data['WavelengthEff'][n]
				flux_zeropoint = data['ZeroPoint'][n]
				zeropoint_unit = data['ZeroPointUnit'][n]
				print(f'{filter_id}: Zero point flux = {flux_zeropoint} {zeropoint_unit}')
					
				# add filter to json file
				# make information a dictionary
				new_filter = {'instrument': fac,
							'filtername': filter_id,
							'lambda_eff': float(lambda_eff),
							'fzeropoint': float(flux_zeropoint)}
				# append new filter to master_filter dictionary
				master_filter.append(new_filter)
				with open(filter_infile, 'w') as json_file:
					json.dump(master_filter, json_file, indent=3)

    
    			# create the filter transmission curve file
				url = url_curve + fac + '/' + filter_id
				resp = requests.get(url)
    			
				outfname = filter_curves_path + fac + '_' + filter_id + '.txt'
    
				with open(outfname, 'wb') as outfile:
					outfile.write(resp.content)
