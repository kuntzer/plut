import cPickle as pickle
import gzip
import os
from astropy.table import Table, vstack, join
import numpy as np
import copy

def writepickle(obj, filepath, protocol = -1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')
	
	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	
def readpickle(filepath):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath,'rb')
	else:
		pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	return obj

def colnorm(cat, name, oname=None):
	"""
	Normalises the column `name` of astropy table `cat`. Optionnally outputs the normalised values to column `oname`. 
	"""
	min_ = cat[name].min()
	max_ = cat[name].max()
	
	col = (cat[name] - min_) / (max_ - min_)
	
	if oname is None:
		cat[name] = col
	else:
		cat[oname] = col
		
def load_truth_cats(dataset):

	if dataset == "train":
		nds = 5
	elif dataset == "test":
		nds = 10

	truth_data = None
	for itrain in range(nds):
		truth_tmp = Table.read("truth_files/catalogs_truth_{}/{}_{:03}_truth_cat.fits".format(dataset, dataset, itrain))
		if truth_data is None:
			truth_data = truth_tmp
		else:
			truth_data = vstack([truth_data, truth_tmp])
			
	return truth_data

def load_encodings(workdir, dataset, snr, calib=False):
	
	if dataset == "train":
		nds = 5
	elif dataset == "test":
		nds = 10
	
	if calib:
		note = "_calib"
	else:
		note = ""
	
	data = None
	for ii in range(nds):
		data_tmp = readpickle(os.path.join(workdir, "inputs", "snr_{}_{}{}_{:03}.pkl".format(snr, dataset, note, ii)))
		if data is None:
			data = data_tmp 
		else:
			data = np.vstack([data, data_tmp])
			
	return data

def merge_code_truth_cats(code, truth):
	
	ids = code["xfield", "yfield"] == truth["xfield", "yfield"]
	# Stupid test to check that we are aligned
	for ii, iids in enumerate(ids):
		if not ids[ii]: print  ii, "FALSE"; raise IndexError()
	#train_truth_data.sort(["xfield", "yfield"])
	#training_data.sort(["xfield", "yfield"])
	
	return join(code, truth, keys=['xfield', 'yfield'])




