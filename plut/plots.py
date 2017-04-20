import numpy as np
import os
import pylab as plt
import matplotlib.ticker as ticker

import metrics, figures, table

def make_condbias_plot(cats, names_components, workdir, train_snr, test_snr, ncbins=5, show=True):
	
	param_feats_field = [
			"g1", "g2", "fwhm", "xfield", "yfield"]
	param_feats_comp = names_components
	
	for iicat, cat in enumerate(cats):
		
		master_Q = metrics.get_Q(cat["delta_g1"], cat["delta_g2"], cat["delta_fwhm"])
		
		if iicat == 0:
			namecat = "test"
		else:
			namecat = "calib"
	
		for iip, param_feats in enumerate([param_feats_field, param_feats_comp]):
		
			isubfig = 1
		
			#nlines = int(np.ceil(2 / (ncol*1.)))
			if iip == 0:
				nameparamfeat = "spatial"
				ncol = 3
				nlines = int(np.ceil(len(param_feats) / (ncol*1.)))
				fig = plt.figure(figsize=(12, 3 * nlines))
			else:
				nameparamfeat = "components"
				ncol = 4
				nlines = int(np.ceil(len(param_feats) / (ncol*1.)))
				fig = plt.figure(figsize=(16, 3 * nlines))
			plt.subplots_adjust(wspace=0.25)
			plt.subplots_adjust(hspace=0.3)
			plt.subplots_adjust(right=0.98)
			plt.subplots_adjust(top=0.98)
			plt.subplots_adjust(left=0.07)
			
			coln = 0
			for iplot, featc in enumerate(param_feats):
				
				ax = fig.add_subplot(nlines, ncol, isubfig + iplot)
				
				ax.set_ylabel(r"Q score")
				
				
				cbinlims = np.array([np.percentile(cat[featc], q) for q in np.linspace(0.0, 100.0, ncbins+1)])
				
				cbinlows = cbinlims[0:-1]
				cbinhighs = cbinlims[1:]
				cbincenters = 0.5 * (cbinlows + cbinhighs)
				assert len(cbincenters) == ncbins
				
				
				Qs = []
				for i in range(ncbins):
					
					# We build the subset of data that is in this color bin:
					selcbin = table.Selector(featc, [("in", featc, cbinlows[i], cbinhighs[i])])
					cbindata = selcbin.select(cat)
					if len(cbindata) == 0:
						continue
					
					de1 = cbindata["delta_g1"]
					de2 = cbindata["delta_g2"]
					dfwhm = cbindata["delta_fwhm"]
					Qs.append(metrics.get_Q(de1, de2, dfwhm))
				
				ax.axhline(master_Q, ls="--", c='grey')
				ax.plot(cbincenters, Qs, lw=2, c='k')	
	
				ax.set_xlabel(featc)
				
				xlims = ax.get_xlim()
				tick_spacing = (xlims[1] - xlims[0]) / 5.
				ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
					
				if coln > 0:
					ax.set_ylabel("")
			
				coln += 1
				if coln == ncol: coln = 0
				
			
			fname = os.path.join(workdir, "plots", "condbias_{}_{}_{}_{}".format(namecat, nameparamfeat, train_snr, test_snr))
			figures.savefig(fname, fig, fancy=True, pdf_transparence=True)
	if show:
		plt.show()