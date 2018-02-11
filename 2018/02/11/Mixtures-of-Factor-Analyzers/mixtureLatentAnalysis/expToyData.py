# Data Generation for Toy Example
from mixture_latent_analysis import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from helper import guassianPlot2D

# z ~ N(0,I), x|z ~ N(Wz+mu, sigma2*I)
def genData4PPCA(num_pts):
	z = np.array([np.random.normal(loc=0, scale=1, size=num_pts)])  # 1xnum_pts
	W = np.array([[3],[1]])  # 2x1
	W = W/np.sqrt(np.sum(W*W))
	mu = np.array([[-2],[1]])  # 2x1
	line_points = np.matmul(W,[[-2, 2]])+mu
	sigma = 0.4
	noise = np.array([np.random.normal(loc=0, scale=sigma, size=num_pts) for i in list(range(2))])
	out = np.matmul(W,z) + mu + noise

	print('gt W={}'.format(W))
	print('gt sigma2={}'.format(sigma*sigma))
	return out, line_points

if __name__ == '__main__':
	num_pts = 400
	data, line_points = genData4PPCA(num_pts)

	# W, mu, sigma2 = ppca_closed_form(data,1)
	# W = W/np.sqrt(np.sum(W*W))
	# print('ppca W={}'.format(W))
	# print('ppca sigma2={}'.format(sigma2))
	# line_points_ppca = np.matmul(W,[[-2, 2]])+mu

	# Return:
	# 	mu, (D,1)
	# 	W, (D,K)
	# 	psi, (D,)
	W, mu, sigma2 = ppca_em(data,1)
	C = np.matmul(W,W.T) + sigma2*np.eye(2)
	W = W/np.sqrt(np.sum(W*W))
	print('ppca_em W={}'.format(W))
	print('ppca_em sigma2={}'.format(sigma2))
	line_points_ppca_em = np.matmul(W,[[-2, 2]])+mu

	W_fa, mu_fa, psi = fa_em(data,1)
	C_fa = np.matmul(W_fa,W_fa.T) + np.diag(psi)
	W_fa = W_fa/np.sqrt(np.sum(W_fa*W_fa))
	print('fa_em W_fa={}'.format(W_fa))
	print('fa_em psi={}'.format(psi))
	line_points_fa_em = np.matmul(W_fa,[[-2, 2]])+mu_fa

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(line_points[0,:],line_points[1,:],'k-')
	# plt.plot(line_points_ppca[0,:],line_points_ppca[1,:],'r--')
	plt.plot(line_points_ppca_em[0,:],line_points_ppca_em[1,:],'g-.')
	pts = guassianPlot2D(mu,C)
	plt.plot(pts[0,:],pts[1,:],'g-.')
	plt.plot(line_points_fa_em[0,:],line_points_fa_em[1,:],'r--')
	pts_fa = guassianPlot2D(mu_fa,C_fa)
	plt.plot(pts_fa[0,:],pts_fa[1,:],'r--')

	plt.legend(['True latent space','PPCA_EM latent space','PPCA_EM marginal pdf','FA_EM latent space','FA_EM marginal pdf'])
	# plt.legend(['True latent space','PPCA latent space','PPCA_EM latent space','FA_EM latent space'])
	ax.scatter(data[0],data[1], c='b', marker='o')

	plt.show()