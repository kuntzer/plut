import numpy as np

def get_Q(error_ell1, error_ell2, error_size, target_error_ell=2e-4, target_error_size=1e-3, eta=2000, sigmamin=1., avg=True):
	"""
	Computes the metrcis. eta and sigmamin are such that if right on target, we have Q=1000
	"""
	
	Qden = (error_ell1 / target_error_ell)**2 + (error_ell2 / target_error_ell)**2 + (error_size / target_error_size)**2
	if avg:
		Qden = np.mean(Qden)
	
	Qden += sigmamin * sigmamin
	
	Q = eta / np.sqrt(Qden)
	
	return Q

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import matplotlib
	import figures
	
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'
	figures.set_fancy()
	
	error_ell = np.logspace(-5, -2, 100)
	error_size = np.logspace(-4, -1, 100)
	
	X, Y = np.meshgrid(error_ell, error_size)
	score = np.zeros_like(X)
	
	score = get_Q(X, X, Y, avg=False)

	fig = plt.figure()
	cs = plt.contourf(X, Y, score, 20, cmap=plt.get_cmap("PuBu_r"), vmin=0, vmax=2000)
	cbar = fig.colorbar(cs)
	
	#CS = plt.contour(X, Y, score, [250,500,1000,1250,1500,1750, 1900])#, colors='r', linestyles="--", linewidths=2)#, [1e3])
	CS = plt.contour(X, Y, score, [1000], colors='r', linestyles="--", linewidths=2)
	
	plt.xscale("log")
	plt.yscale("log")
	
	plt.xlabel(r"$\langle e_\mathrm{obs} - e_\mathrm{true} \rangle$")
	plt.ylabel(r"$\langle R_\mathrm{obs}^2 - R_\mathrm{true}^2 \rangle/\langle R_\mathrm{true}^2\rangle$")
	
	plt.tight_layout()
	figures.savefig("Qcontours", fig, fancy=True, pdf_transparence=True)
	plt.show()
