import os
import numpy as np
import tenbilac
from astropy.table import Table
import astropy.table
import pylab as plt


import plut 

import logging

logging.basicConfig(format='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)
###################################################################################################
# Variables 

workdir = "PCA"

n_components = 8

do_train = False		
do_self_predict = False
do_predict = True

tenbilacconfigname = "sum55.cfg"

train_dataset = "train"
train_snr = "no"

test_dataset = "test"
test_snr = "no"#train_snr

show = True

###################################################################################################
# Initialisation: loading the files

training_data = plut.utils.load_encodings(workdir, train_dataset, train_snr)

assert np.size(np.unique(training_data[:,5])) == np.shape(training_data)[0]
assert np.size(np.unique(training_data[:,6])) == np.shape(training_data)[0]

names_components = ["component_{:02d}".format(ii) for ii in range(n_components)] 
names_cols = ["xfield", "yfield"] + names_components
#dtypes = [np.int64, np.int64, np.float64, np.float64]
#dtypes += [np.float64 for n in range(n_components)]
training_data = Table(training_data, names=names_cols)#, dtype=dtypes)

if show:
	plt.figure(figsize=(16, 6))

	for n in range(n_components):
		ax = plt.subplot(2, 4, n+1)
		hh = training_data["component_{:02}".format(n)]
		cs = ax.scatter(training_data["xfield"], training_data["yfield"], c=hh, edgecolor="None")
		plt.colorbar(cs)
		ax.set_title("PCA - Component {}".format(n+1))
	plt.tight_layout(h_pad=0.1)
	plt.show()

train_truth_data = plut.utils.load_truth_cats(train_dataset)

joined_cats = plut.utils.merge_code_truth_cats(training_data, train_truth_data)

#--------------------------------------------------------------------------------------------------

test_data = plut.utils.load_encodings(workdir, test_dataset, test_snr)
print test_data.shape
test_data = Table(test_data, names=names_cols)
test_truth = plut.utils.load_truth_cats(test_dataset)
test_joined_cats = plut.utils.merge_code_truth_cats(test_data, test_truth)


test_calib_data = plut.utils.load_encodings(workdir, test_dataset, test_snr, calib=True)
test_calib_data = Table(test_calib_data, names=names_cols)
test_calib_joined_cats = plut.utils.merge_code_truth_cats(test_calib_data, test_truth)

#--------------------------------------------------------------------------------------------------
toolconfpath = os.path.join(workdir, "configs", tenbilacconfigname)
toolconfig = plut.ml.readconfig(toolconfpath) # The Tenbilac config
confname = toolconfig.get("setup", "name") # Will be passed to Tenbilac

if show:
	hh = joined_cats["component_07"]
	plt.figure()
	plt.scatter(joined_cats["xfield"], joined_cats["yfield"], c=hh, edgecolor="None", s=5)
	plt.colorbar()
	
	plt.show()

###################################################################################################
# Learning the mapping components, self-prediction and prediction on data set

inputlabels = names_components

for name in ["g1", "g2", "fwhm"]:
	trainworkdir = os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), name)
	tblconfiglist = [("setup", "workdir", trainworkdir), ("setup", "name", confname)]
	
	ten = tenbilac.com.Tenbilac(os.path.join(workdir, "configs", tenbilacconfigname), tblconfiglist)
	targetlabels = [name]
	predlabels = ["meas_{}".format(name)]
		
	if do_train:
		plut.ml.train(ten, joined_cats, inputlabels, targetlabels)
	
	if do_self_predict:
		joined_cats = plut.ml.predict(ten, joined_cats, inputlabels, predlabels)
			
		joined_cats.write(os.path.join(trainworkdir, "selfpredcat.fits"), overwrite=True)
	else:
		joined_cats = Table.read(os.path.join(trainworkdir, "selfpredcat.fits"))

	if show:
		print "Components are fitted for {}. Moving on.".format(name)
		plt.figure()
		plt.scatter(joined_cats[name], joined_cats['meas_{}'.format(name)]-joined_cats[name])
		plt.show()

###################################################################################################
# Predict the test set 

if do_predict:
	for name in ["g1", "g2", "fwhm"]:
		predlabels = ["meas_{}".format(name)]
		trainworkdir = os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), name)
		tblconfiglist = [("setup", "workdir", trainworkdir), ("setup", "name", confname)]
		
		ten = tenbilac.com.Tenbilac(os.path.join(workdir, "configs", tenbilacconfigname), tblconfiglist)
		test_joined_cats = plut.ml.predict(ten, test_joined_cats, inputlabels, predlabels)
		
		test_joined_cats["delta_{}".format(name)] = test_joined_cats["meas_{}".format(name)] - test_joined_cats["{}".format(name)]
		
		test_calib_joined_cats = plut.ml.predict(ten, test_calib_joined_cats, inputlabels, predlabels)
		
		test_calib_joined_cats["delta_{}".format(name)] = test_calib_joined_cats["meas_{}".format(name)] - test_calib_joined_cats["{}".format(name)]  

	test_joined_cats["delta_fwhm"] /= np.mean(test_joined_cats["fwhm"])		
	test_joined_cats.write(os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), "valpredcat.fits"), overwrite=True)
	
	test_calib_joined_cats["delta_fwhm"] /= np.mean(test_calib_joined_cats["fwhm"])		
	test_calib_joined_cats.write(os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), "valpredcat_calib.fits"), overwrite=True)
else:
	test_joined_cats = Table.read(os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), "valpredcat.fits"))
	test_calib_joined_cats = Table.read(os.path.join(workdir, "fit", "{}_{}".format(train_dataset, train_snr), "valpredcat_calib.fits"))

###################################################################################################
# Evaluate the Q metrics

print "std de1: %1.2e" % np.std(test_joined_cats["delta_g1"]), "rel: %1.2e" %  (np.std(test_joined_cats["delta_g1"]) / 2e-4)
print "std de2: %1.2e" % np.std(test_joined_cats["delta_g2"]), "rel: %1.2e" %  (np.std(test_joined_cats["delta_g2"]) / 2e-4)
print "std dFWHM/FWHM0: %1.2e" % np.std(test_joined_cats["delta_fwhm"]), "rel: %1.2e" % (np.std(test_joined_cats["delta_fwhm"]) / 1e-3)
print "Q Compression: {:4.1f}".format( plut.metrics.get_Q(test_calib_joined_cats["delta_g1"], test_calib_joined_cats["delta_g2"], test_calib_joined_cats["delta_fwhm"]))
master_Q = plut.metrics.get_Q(test_joined_cats["delta_g1"], test_joined_cats["delta_g2"], test_joined_cats["delta_fwhm"])
print "Q Compression+Interpolation: {:4.1f}".format(master_Q)
	
	
###################################################################################################
# Compute conditional metrics 

cats = [test_joined_cats, test_calib_joined_cats]
plut.plots.make_condbias_plot(cats, names_components, workdir, train_snr, test_snr, show=True)