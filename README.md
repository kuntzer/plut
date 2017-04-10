# plut

# Requirements:

You need to install [Astropy](http://astropy.readthedocs.io) and [Tenbilac](https://github.com/mtewes/tenbilac)

# To run:

Run `python 41_encode_pca.py` for PIBE and copy/paste the encoded representation of the PCA in `PCA/inputs`. 
Then `python test_pca.py` to fit Tenbilac through the results and compute the metrics.

A typical run should yield:

* Training on SNR=no
	* Test on SNR=no, Q=1934
	* Test on SNR=100, Q=18
	* Test on SNR=20, Q=19

* Training on SNR=100
	* Test on SNR=no, Q=18
	* Test on SNR=100, Q=1578
	* Test on SNR=20, Q=54

* Training on SNR=20
	* Test on SNR=no, Q=17
	* Test on SNR=100, Q=16
	* Test on SNR=20, Q=570

This simple PCA run requires that the noise in training should be the same as in the test set.
