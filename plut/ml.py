import datetime
import numpy as np
import astropy
import astropy.table
import copy

import os
from ConfigParser import SafeConfigParser

import tenbilac

import logging
logger = logging.getLogger(__name__)

def readconfig(configpath):
	"""
	Reads in a config file
	"""
	config = SafeConfigParser(allow_no_value=True)
	
	if not os.path.exists(configpath):
		raise RuntimeError("Config file '{}' does not exist!".format(configpath))
	logger.debug("Reading config from '{}'...".format(configpath))
	config.read(configpath)
	
	name = config.get("setup", "name") 
	if name is None or len(name.strip()) == 0: # if the ":" is missing as well, confirparser reads None
		# Then we use the filename
		config.set("setup", "name", os.path.splitext(os.path.basename(configpath))[0])
	logger.info("Read config '{}' from file '{}'.".format(config.get("setup", "name"), configpath))	
	
	return config

def get3Ddata(catalog, colnames):
	"""
	Function to build a 3D numpy array (typically for Tenbilac input) from some columns of an astropy catalog.
	The point is to ensure that all columns get the same shape.
	
	The 3D output array has shape (realization, feature, case).
	"""
	
	if len(colnames) == 0:
		raise RuntimeError("No colnames to get data from!")
	
	# Check for exotic catalogs (do they even exist ?)
	for colname in colnames:
		if not catalog[colname].ndim in [1, 2]:
			raise RuntimeError("Can only work with 1D or 2D columns")
	
	# Let's check the depths of the 2D colums to see what size we need.
	nreas = list(set([catalog[colname].shape[1] for colname in colnames if catalog[colname].ndim == 2]))
	#logger.info("We have the following nreas different from one in there: {}".format(nreas))
	if len(nreas) > 1:
		raise RuntimeError("The columns have incompatible depths!")

	if len(nreas) == 0:
		nrea = 1
		logger.info("For each column, only one realization is available.")
		
	else:
		nrea = nreas[0]
		logger.info("Extracting data from {0} realizations...".format(nrea))
		nrea = nreas[0]
	
	if "ngroup" in catalog.meta:
		if nrea != catalog.meta["ngroup"]:
			raise RuntimeError("Something very fishy: depth is not ngroup!")

	# And now we get the data:
	
	readycols = []
	for colname in colnames:
				
		col = np.ma.array(catalog[colname])
				
		if col.ndim == 2:
			pass
			
		elif col.ndim == 1:
			# This column has only one realization, and we have to "duplicate" it nrea times...
			col = np.tile(col, (nrea, 1)).transpose()
					
		else:
			raise RuntimeError("Weird column dimension")
								
		readycols.append(col)
		
	outarray = np.rollaxis(np.ma.array(readycols), 2)
	
	assert outarray.ndim == 3
	assert outarray.shape[1] == len(colnames)

	return outarray

def train(ten, joined_cats, inputlabels, targetlabels):
	inputsdata = get3Ddata(joined_cats, inputlabels)
	
	# Preparing the targets
	for colname in targetlabels:
		if not np.all(np.logical_not(np.ma.getmaskarray(joined_cats[colname]))): # No element should be masked.
			raise RuntimeError("Targets should not be masked, but '{}' is!".format(colname))
	targetsdata = np.column_stack([np.array(joined_cats[colname]) for colname in targetlabels]).transpose()
	assert inputsdata.shape[2] == targetsdata.shape[1] # Number of cases should match
	
	#print plut.ml.get3Ddata(indata, inputlabels)
	ten.train(inputsdata, targetsdata, inputlabels, targetlabels)

def predict(ten, joined_cats, inputlabels, predlabels):
	
	inputsdata = get3Ddata(joined_cats, inputlabels)
	preddata = ten.predict(inputsdata)
	
	for (i, predlabel) in enumerate(predlabels):	
		logger.info("Adding predictions '{}' to catalog...".format(predlabel))
		data = preddata[:,i,:].transpose()
		assert data.ndim == 2 # Indeed this is now always 2D.
		if data.shape[1] == 1: # If we have only one realization, just make it a 1D numpy array.
			data = data.reshape((data.size))
			assert data.ndim == 1
						
		newcol = astropy.table.MaskedColumn(data=data, name=predlabel)
		joined_cats.add_column(newcol)
		
	return joined_cats
			