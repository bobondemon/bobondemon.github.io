# Data Generation for Toy Example
from mixture_latent_analysis import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from helper import guassianPlot2D

# z ~ N(0,I), x|z ~ N(Wz+mu, sigma2*I)
def genData4MFA(num_pts):
	pi1 = 0.25
	num_pts1 = int(pi1*num_pts)
	z = np.array([np.random.normal(loc=0, scale=1, size=num_pts1)])  # 1xnum_pts
	W1 = np.array([[-3],[1]])  # 2x1
	W1 = W1/np.sqrt(np.sum(W1*W1))
	mu1 = np.array([[-2],[1]])  # 2x1
	line_points1 = np.matmul(W1,[[-2, 2]])+mu1
	sigma1 = 0.4
	noise1 = np.array([np.random.normal(loc=0, scale=sigma1, size=num_pts1) for i in list(range(2))])
	out1 = np.matmul(W1,z) + mu1 + noise1  # 2xnum_pts

	pi2 = 1.0-pi1
	num_pts2 = int(pi2*num_pts)
	z = np.array([np.random.normal(loc=0, scale=1, size=num_pts2)])  # 1xnum_pts
	W2 = np.array([[1.5],[2.2]])  # 2x1
	W2 = W2/np.sqrt(np.sum(W2*W2))
	# mu2 = np.array([[10],[-2]])  # 2x1
	mu2 = np.array([[1.5],[-2]])  # 2x1, ERROR, "array must not contain infs or NaNs" in multivariate_normal
	line_points2 = np.matmul(W2,[[-2, 2]])+mu2
	sigma2 = 0.4
	noise2 = np.array([np.random.normal(loc=0, scale=sigma2, size=num_pts2) for i in list(range(2))])
	out2 = np.matmul(W2,z) + mu2 + noise2  # 2xnum_pts

	return out1, line_points1, out2, line_points2

if __name__ == '__main__':
	num_pts = 400
	data1, line_points1_orig, data2, line_points2_orig = genData4MFA(num_pts)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.title(['Cluster1 with {} of pts'.format(len(data1[0])),'Cluster2 with {} of pts'.format(len(data2[0]))])

	X = np.concatenate([data1,data2],axis=1)
	# 	pi, (M,)
	# 	mu, (M,D)
	# 	W, (M,D,K)
	# 	psi, (D,)
	pi, mu, W, psi = mfa_em(X,K=1,M=2)

	print('pi1={}; pi2={}'.format(pi[0],pi[1]))
	print('mu1={}; mu2={}'.format(mu[0,:],mu[1,:]))

	C1 = np.matmul(W[0,...],W[0,...].T) + np.diag(psi)
	C2 = np.matmul(W[1,...],W[1,...].T) + np.diag(psi)

	print('C1={}; C2={}'.format(C1,C2))

	W[0,...] = W[0,...]/np.sqrt(np.sum(W[0,...]*W[0,...]))
	W[1,...] = W[1,...]/np.sqrt(np.sum(W[1,...]*W[1,...]))
	line_points1 = np.matmul(W[0,...],[[-2, 2]]) + mu[0,:].reshape((2,1))
	line_points2 = np.matmul(W[1,...],[[-2, 2]]) + mu[1,:].reshape((2,1))

	pts1 = guassianPlot2D(mu[0,...],C1)
	pts2 = guassianPlot2D(mu[1,...],C2)

	plt.plot(pts1[0,:],pts1[1,:],'g-',linewidth=3)
	plt.plot(pts2[0,:],pts2[1,:],'y-',linewidth=3)

	plt.legend(['MFA1','MFA2'])

	plt.plot(line_points1[0,:],line_points1[1,:],'g-',linewidth=2)
	plt.plot(line_points2[0,:],line_points2[1,:],'y-',linewidth=2)

	ax.scatter(data1[0],data1[1], c='b', marker='o')
	ax.scatter(data2[0],data2[1], c='r', marker='o')

	plt.plot(line_points1_orig[0,:],line_points1_orig[1,:],'k-')
	plt.plot(line_points2_orig[0,:],line_points2_orig[1,:],'k-')

	plt.show()