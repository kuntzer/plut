import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import plut

white = True

if white:
	import matplotlib.font_manager as fm
	from matplotlib import rc
	prop = fm.FontProperties(fname='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreadventor-regular.otf')
	rc('font', **{'family':'TeX Gyre Adventor','size':14})
	matplotlib.rcParams.update({'text.color': 'white', 'ytick.color':'white', 'xtick.color':'white',
							'axes.labelcolor':'white'})

N = 10000
step = 500#250
errs = []
for nout in range(N+step):
	if not (nout % step) == 0: continue 
	
	e1out = np.random.normal(scale=4.5e-4, size=nout)
	e2out = np.random.normal(scale=4e-4, size=nout)
	Rout = np.random.normal(scale=2e-3, size=nout)
	
	scores = []
	for i in range(1000):
		e1 = np.random.normal(scale=2e-4, size=N)
		e2 = np.random.normal(scale=2e-4, size=N)
		R = np.random.normal(scale=1e-3, size=N)
		
		e1[:nout] = e1out
		e2[:nout] = e2out
		R[:nout] = Rout
		
		score = plut.metrics.get_Q(e1, e2, R)
		scores.append(score)
	
	scores = np.asarray(scores)
	err = [float(nout)/N*1e2, scores.mean(), scores.std()]
	print err
	errs.append(err)
errs = np.array(errs)
	
fig = plt.figure()
ax = plt.subplot(111)
if white:
	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white') 
	ax.spines['right'].set_color('white')
	ax.spines['left'].set_color('white')
	c = 'w'
else:
	c = 'k'
plt.plot(errs[:,0], errs[:,1], lw=2, c=c)
plt.xlabel("Percentage of outliers")
plt.ylabel("Q score")
plt.grid()
plt.ylim([500,1000])
plt.tight_layout()
plut.figures.savefig("stresstest", fig, fancy=True, pdf_transparence=True)
plt.show()