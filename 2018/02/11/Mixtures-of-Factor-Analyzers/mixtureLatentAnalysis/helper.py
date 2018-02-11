from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import *

def guassianPlot2D(mu,sigma):
	mu = np.array(mu).reshape((-1,1))
	sigma = np.array(sigma).reshape(2,2)
	assert(len(mu)==2)

	# first we generate the unit circle of (x,y) points
	def PointsInCircum(r,n=100):
		return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]
	pts = np.array(PointsInCircum(2)).T  # 2xN

	# we then calculate the sqrt of the sigma
	# the np.eig has output ( (N,), (N,N) )
	# where each col of eig_vec is eigen vector and is of norm=1
	# note that eig_val is not sorted
	# eig_vec * eig_val * eig_vec.T = sigma
	eig_val, eig_vec = eig(sigma)  # we assume sigma is positive definite, so eig_val > 0
	eig_val_sqrt = np.sqrt(eig_val).reshape((1,-1))  # 1x2
	sigma_sqrt = eig_vec*eig_val_sqrt  # 2x2

	# finally, transform pts based on sigma_sqrt
	# y = Ax, cov(y) = A*cov(x)*A.T
	# since cov(x) = I, so cov(y) = A*A.T
	# if we let A = sqrt(sigma), then cov(y) = sigma, which is the covariance matrix we need
	pts = np.matmul(sigma_sqrt,pts)  # 2xN
	pts += mu

	return pts

# Assume X is of size DxN
# Return an (N,) array
def calGassuianProb(X,mu,cov):
	mu = np.array(mu).reshape((-1,1))  # Dx1
	D = len(mu)
	cov = np.array(cov).reshape(D,D)
	X_mu = X - mu  # DxN
	return np.exp ( - (np.log(np.linalg.det(cov)) + np.diag(np.matmul(np.matmul(X_mu.T,inv(cov)),X_mu)) + D*np.log(2*np.pi)) / 2 )

