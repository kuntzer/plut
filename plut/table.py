"""
Helpers for astropy.table arrays
"""

import numpy as np
import astropy.table
import datetime

import copy

import logging
logger = logging.getLogger(__name__)


def info(cat, txt=True):
	"""
	Returns a new table "describing" the content of the table cat.
	"""
	colnames = cat.colnames
	dtypes = [cat[colname].dtype for colname in colnames]
	ndims = [cat[colname].ndim for colname in colnames]
	shapes = [cat[colname].shape for colname in colnames]
	infotable = astropy.table.Table([colnames, dtypes, ndims, shapes], names=("colname", "dtype", "ndim", "shape"))
	
	infotable.sort("colname")
	infotable.meta = cat.meta
	
	if txt:
		
		lines = infotable.pformat(max_lines=-1, max_width=-1)
		lines.append("")
		lines.append("Number of rows: {}".format(len(cat)))
		lines.append("Number of columns: {}".format(len(cat.colnames)))
		lines.append("Metadata: {}".format(str(infotable.meta.items())))
		
		return "\n".join(lines)
	else:
		return infotable




class Selector:
	"""
	Provides a simple way of getting "configurable" sub-selections of rows from a table.
	"""
	
	def __init__(self, name, criteria):
		"""
		
		:param name: a short string describing this selector (like "star", "low_snr", ...)
		:param criteria: a list of tuples describing the criteria. Each of these tuples starts 
			with a string giving its type, followed by some arguments.
	
			Illustration of the available criteria (all limits are inclusive):
		
			- ``("in", "tru_rad", 0.5, 0.6)`` : ``"tru_rad"`` is between 0.5 and 0.6 ("in" stands for *interval*) and *not* masked
			- ``("max", "snr", 10.0)`` : ``"snr"`` is below 10.0 and *not* masked
			- ``("min", "adamom_flux", 10.0)`` : ``"adamom_flux"`` is above 10.0 and *not* masked
			- ``("inlist", "subfield", (1, 2, 3))`` : ``subfield`` is among the elements in the tuple or list (1,2,3) and *not* masked.
			- ``("is", "Flag", 2)`` : ``"Flag"`` is exactly 2 and *not* masked
			- ``("nomask", "pre_g1")`` : ``"pre_g1"`` is not masked
			- ``("mask", "snr")`` : ``"snr"`` is masked
		
		
		"""
		self.name = name
		self.criteria = criteria
	
	def __str__(self):
		"""
		A string describing the selector
		"""
		return "'%s' %s" % (self.name, repr(self.criteria))
	
	
	def combine(self, *others):
		"""
		Returns a new selector obtained by merging the current one with one or more others.

		:param others: provide one or several other selectors as arguments.

		.. note:: This does **not** modify the current selector in place! It returns a new one!
		"""
	
		combiname = "&".join([self.name] + [other.name for other in others])
	
		combicriteria = self.criteria
		for other in others:
			combicriteria.extend(other.criteria)
	
		return Selector(combiname, combicriteria)
		
	
	def select(self, cat):
		"""
		Returns a copy of cat with those rows that satisfy all criteria.
		
		:param cat: an astropy table
		
		"""
		
		if len(self.criteria) is 0:
			logger.warning("Selector %s has no criteria!" % (self.name))
			return copy.deepcopy(cat)
		
		passmasks = []
		for crit in self.criteria:
		
			if cat[crit[1]].ndim != 1:
				logger.warning("Selecting with multidimensional column ('{}', shape={})... hopefully you know what you are doing.".format(crit[1], cat[crit[1]].shape))
			
			if crit[0] == "in":
				if len(crit) != 4: raise RuntimeError("Expected 4 elements in criterion %s" % (str(crit)))
				passmask = np.logical_and(cat[crit[1]] >= crit[2], cat[crit[1]] <= crit[3])
				if np.ma.is_masked(passmask):
					passmask = passmask.filled(fill_value=False)
				# Note about the "filled": if crit[2] or crit[3] englobe the values "underneath" the mask,
				# some masked crit[1] will result in a masked "passmask"!
				# But we implicitly want to reject masked values here, hence the filled.
							
			elif crit[0] == "max":
				if len(crit) != 3: raise RuntimeError("Expected 3 elements in criterion %s" % (str(crit)))
				passmask = (cat[crit[1]] <= crit[2])
				if np.ma.is_masked(passmask):
					passmask = passmask.filled(fill_value=False)
			
			elif crit[0] == "min":
				if len(crit) != 3: raise RuntimeError("Expected 3 elements in criterion %s" % (str(crit)))
				passmask = (cat[crit[1]] >= crit[2])
				if np.ma.is_masked(passmask):
					passmask = passmask.filled(fill_value=False)
			
			elif crit[0] == "inlist":
				if len(crit) != 3: raise RuntimeError("Expected 3 elements in criterion %s" % (str(crit)))
				passmask = np.in1d(np.asarray(cat[crit[1]]), crit[2]) # This ignores any mask
				if np.ma.is_masked(passmask): # As the mask is ignored by in1d, this is probably worthless and will never happen
					passmask = passmask.filled(fill_value=False)
				# So we need to deal with masked elements manually:
				if hasattr(cat[crit[1]], "mask"): # i.e., if this column is masked:
					passmask = np.logical_and(passmask, np.logical_not(cat[crit[1]].mask))
						
			elif crit[0] == "is":
				if len(crit) != 3: raise RuntimeError("Expected 3 elements in criterion %s" % (str(crit)))
				passmask = (cat[crit[1]] == crit[2])
				if np.ma.is_masked(passmask):
					passmask = passmask.filled(fill_value=False)
					
			elif (crit[0] == "nomask") or (crit[0] == "mask"):
				if len(crit) != 2: raise RuntimeError("Expected 2 elements in criterion %s" % (str(crit)))
				if hasattr(cat[crit[1]], "mask"): # i.e., if this column is masked:
					if crit[0] == "nomask":
						passmask = np.logical_not(cat[crit[1]].mask)
					else:
						passmask = cat[crit[1]].mask
				else:
					logger.warning("Criterion %s is facing an unmasked column!" % (str(crit)))
					passmask = np.ones(len(cat), dtype=bool)
			
			else:
				raise RuntimeError("Unknown criterion %s" % (crit))
					
			logger.debug("Criterion %s of '%s' selects %i/%i rows (%.2f %%)" %
				(crit, self.name, np.sum(passmask), len(cat), 100.0 * float(np.sum(passmask))/float(len(cat))))
			
			assert len(passmask) == len(cat)
			passmasks.append(passmask) # "True" means "pass" == "keep this"
		
		# Combining the passmasks:
		passmasks = np.logical_not(np.column_stack(passmasks)) # "True" means "reject"
		combimask = np.logical_not(np.sum(passmasks, axis=1).astype(bool)) # ... and "True" means "keep this" again.
		
		logger.info("Selector '%s' selects %i/%i rows (%.2f %%)" %
				(self.name, np.sum(combimask), len(cat), 100.0 * float(np.sum(combimask))/float(len(cat))))
		
		return cat[combimask]

