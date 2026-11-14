import numpy as np
from numpy import matmul, trace
from numpy.linalg import *
from helper import *
from scipy.stats import multivariate_normal

# K is the dim of latent variable z
# z ~ N(0,I), x|z ~ N(Wz+mu, sigma2*I)
# based on linear-Gaussian framework, we have x ~ N(mu,C)
# where C = WW^T + sigma2*I

# X is of dim (DxN), where N is number of data points
# K should be the dim of latent variable z, and supposed to be smaller than D
def ppca_closed_form(X,K):
	X = np.array(X)
	D, N = X.shape
	assert(K<=D and K>0)

	# compute sample mean
	mu = np.mean(X,axis=1).reshape([-1,1])

	# calculate the covariance matrix S
	S = matmul((X-mu),(X-mu).T)/N
	eig_val, eig_vec = eig(S)	# eig is not sorted
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	U = np.array([ eig_pairs[i][1] for i in range(K) ]).T  # DxK
	# print(U.shape)
	L = np.diagflat([ eig_pairs[i][0] for i in range(K) ])  # KxK
	# print(L.shape)
	W = matmul(U,L)  # DxK
	# print(W.shape)

	# comput sigma2, mean of the rest D-K eigenvalues
	sigma2 = np.mean([ eig_pairs[K+i][0] for i in range(D-K) ])

	return W, mu, sigma2

# X is of dim (DxN), where N is number of data points
# K should be the dim of latent variable z, and supposed to be smaller than D
def ppca_em(X,K):
	itr_num = 50
	X = np.array(X)
	D, N = X.shape
	assert(K<=D and K>0)

	# compute sample mean
	mu = np.mean(X,axis=1).reshape([-1,1])
	X_mu = X - mu

	# Initialization for the parameters
	W = np.array(np.random.normal(loc=0, scale=0.1, size=D*K)).reshape([D,K])  # DxK matrix
	sigma2 = 0.1

	for itr in range(itr_num):
		# ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
		# E_(z|x)[z] = inv(M)W'(x_n - mu), where M = W'W + sigma2*I (an KxK matrix)
		invM = inv( matmul(W.T,W) + np.diag([sigma2]*K) )
		Ez = matmul(matmul(invM,W.T),X_mu)  # KxN
		# print('Ez.shape={}'.format(Ez.shape))
		# sumE_(z|x)[zz'] = N*sigma2*invM + sum(Ez*Ez')
		sumEzz = N*sigma2*invM + matmul(Ez,Ez.T)  # KxK
		# print('sumEzz.shape={}'.format(sumEzz.shape))

		# ===== Maximization Stage
		W_new = matmul(matmul(X_mu,Ez.T),inv(sumEzz))  # DxK matrix
		# print('W_new.shape={}'.format(W_new.shape))
		# print('Shape of norm(X_mu)={}'.format(norm(X_mu,axis=0).shape))
		
		term1 = np.sum(X_mu*X_mu)
		term2 = -2*np.sum([ matmul(matmul(Ez.T[i,:],W_new.T),X_mu[:,i]) for i in range(N) ])
		term3 = trace( matmul(matmul(sumEzz,W_new.T),W_new) )
		sigma2_new =  (term1 + term2 + term3)/(N*D)

		# ===== Update Parameters
		W = W_new
		sigma2 = sigma2_new

	return W, mu, sigma2

# X is of dim (DxN), where N is number of data points
# K should be the dim of latent variable z, and supposed to be smaller than D
# Return:
# 	mu, (D,1)
# 	W, (D,K)
# 	psi, (D,)
def fa_em(X,K):
	itr_num = 50
	X = np.array(X)
	D, N = X.shape
	assert(K<=D and K>0)

	# compute sample mean
	mu = np.mean(X,axis=1).reshape([-1,1])
	X_mu = X - mu  # DxN matrix
	# compute the data covariance matrix
	S = matmul(X_mu,X_mu.T)/N

	# Initialization for the parameters
	W = np.array(np.random.normal(loc=0, scale=0.1, size=D*K)).reshape([D,K])  # DxK matrix, interpreted as factor loading
	psi = np.array([0.1]*D).reshape((1,-1))  # 1xD matrix, interpreted as uniqueness

	for itr in range(itr_num):
		# ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
		# E_(z|x)[z] = G*W'*inv(psi)*(x_n - mu), where G = inv(I + W'*inv(psi)*W)
		G = inv( np.eye(K) + matmul(W.T/psi,W) )  # KxK matrix
		Ez = matmul(matmul(G,W.T)/psi,X_mu)  # KxN
		# print('Ez.shape={}'.format(Ez.shape))
		# sumE_(z|x)[zz'] = N*G + sum(Ez*Ez')
		sumEzz = N*G + matmul(Ez,Ez.T)  # KxK
		# print('sumEzz.shape={}'.format(sumEzz.shape))

		# ===== Maximization Stage
		W_new = matmul(matmul(X_mu,Ez.T),inv(sumEzz))  # DxK matrix
		# print('W_new.shape={}'.format(W_new.shape))
		# print('Shape of norm(X_mu)={}'.format(norm(X_mu,axis=0).shape))
		
		psi_new = np.diag( S - matmul(W_new, matmul(Ez,X_mu.T)/N ) ).reshape((1,-1))

		# ===== Update Parameters
		W = W_new
		psi = psi_new

	return W, mu, psi[0]


# "The EM Algorithm for Mixtures of Factor Analyzers", Zoubin Ghahramani and Geoffrey E. Hinton
# where the notations still follow ML book
# X is of dim (DxN), where N is number of data points
# K should be the dim of latent variable z, and supposed to be smaller than D
# M should be the number of mixtures, which must less than N
# We have the following relationship
# P(x|z,mixture_j) ~ N(mu[mixture_j,:]+W[mixture_j,:,:]z, psi)
# and each mixture has there weight pdf as P(mixture_j) = pi[mixture_j]
# Return:
# 	pi, (M,)
# 	mu, (M,D)
# 	W, (M,D,K)
# 	psi, (D,)
def mfa_em(X,K,M):
	itr_num = 30
	X = np.array(X)
	D, N = X.shape
	assert(K<=D and K>0)
	assert(M<=N and M>0)

	# Initialization for the parameters
	pi = np.array([1]*M)/M  # (M,) array
	# We should have a good initialize of mu.
	# By doing so, we can avoid dividing zero when probs of all samples are approximately close to 0 for a bad initialized mixutre
	mu = np.array(np.random.normal(loc=0, scale=0.1, size=M*D)).reshape([M,D,1])  # MxDx1 matrix, each row represents an mean vector for a mixture
	W = np.array(np.random.normal(loc=0, scale=0.1, size=M*D*K)).reshape([M,D,K])  # MxDxK matrix, interpreted as factor loading
	psi = np.array([0.1]*D).reshape((1,-1))  # 1xD matrix, interpreted as uniqueness, this parameter is SHARED for all mixtures

	for itr in range(itr_num):
		# ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
		# H is of size MxN, where hij is the prob of xj been in mixture j
		H = np.zeros((M,N))
		for j in range(M):
			covMat = np.matmul(W[j,...],W[j,...].T) + np.diag(psi[0])
			# print('covMat={}'.format(covMat))
			# multiGaussian = multivariate_normal(mu[j,...].reshape(-1), covMat)
			# tmp = multiGaussian.pdf(X.T)
			# Avoiding a bad mixture that has prob=0 for all samples
			badMixture = True
			while badMixture:
				tmp2 = calGassuianProb(X,mu[j,...].reshape(-1), covMat)  # (N,) array
				# print('tmp={}'.format(tmp))
				# print('tmp2={}'.format(tmp2))
				# print('pdf.shape={}'.format(tmp.shape))
				if np.sum(tmp2)<=1e-15:
					new_mixture_mean = X[:,np.random.randint(N)]
					print('BAD {}-th Gaussian component!\n Try to Re-initialize with new mean = {}'.format(j,new_mixture_mean))
					mu[j,:,0] = new_mixture_mean
				else:
					badMixture = False

			H[j,:] = tmp2
			# print('H[{},:].shape={}'.format(j,H[j,:].shape))
		# print('np.sum(H,axis=0)={}'.format(np.sum(H,axis=0)))
		# Avoiding bad sample that has prob=0 for all mixtures, if it happens, set equal prob.
		sumH = np.sum(H,axis=0)  # (N,)
		zero_idx = [ i for (i,v) in enumerate(sumH) if v<1e-15]
		H[:,zero_idx] += np.array([1.0]*D).reshape((-1,1))/D
		sumH[zero_idx] = 1
		H = H/sumH.reshape((1,-1))  # Should avoid divid by zero

		X_mu = []  # MxDxN matrix
		for j in range(M):
			X_mu.append(X - mu[j,...])
		X_mu = np.array(X_mu)

		# we use j to represent the index of which mixture (mixture_j)
		# E_(j,z|x)[j,z] = Gj*Wj'*inv(psi)*(x_n - muj) * Hj, where Gj = inv(I + Wj'*inv(psi)*Wj)
		# sumE_(j,z|x)[j,zz'] = N*Gj + sum(Ez[j,...]*Ez[j,...]')
		Ez = np.zeros((M,K,N))  # MxKxN
		sumEzz = np.zeros((M,K,K))  # MxKxK
		for j in range(M):
			# print('psi.shape={}'.format(psi.shape))
			# print('W[j,...].shape={}'.format(W[j,...].shape))
			# print('shape of np.eye(K)+matmul(W[j,...].T/psi,W[j,...]) = {}'.format( (np.eye(K)+matmul(W[j,...].T/psi,W[j,...])).shape) )
			Gj = inv( np.eye(K) + matmul(W[j,...].T/psi,W[j,...]) )  # KxK matrix
			Hj = H[j,:].reshape((1,N))  # (1,N) matrix
			Eztmp = matmul(matmul(Gj,W[j,...].T)/psi,X_mu[j,...])  # KxN
			Ez[j,...] = Eztmp * Hj  # KxN

			sumEzz[j,...] = np.sum(Hj)*Gj + matmul(Eztmp*Hj,Eztmp.T)  # KxK

		# ===== Maximization Stage
		# Update pi
		pi_new = np.sum(H,axis=1)/N  # (M,) array
		# print('pi_new={}'.format(pi_new))
		# print('H.shape={}'.format(H.shape))
		# print(H)
		# print(np.sum(H,axis=0))
		# Update mu, MxDx1 matrix
		mu_new = np.zeros_like(mu)
		for j in range(M):
			Hj = H[j,:].reshape((1,N))  # (1,N) matrix
			# X is of size DxN
			mu_new[j,...] = (np.sum(X*Hj,axis=1)/np.sum(Hj)).reshape((-1,1))
		# Update W, MxDxK matrix
		W_new = np.zeros_like(W)
		for j in range(M):
			# Hj = H[j,:].reshape((1,N))  # (1,N) matrix
			# X_mu[j,...] DxN matrix
			W_new[j,...] = matmul(matmul(X_mu[j,...],Ez[j,...].T),inv(sumEzz[j,...]))  # DxK matrix
		# Update psi, 1xD matrix
		psi_new = np.zeros_like(psi)  # 1xD matrix
		for j in range(M):
			Hj = H[j,:].reshape((1,N))  # (1,N) matrix
			# X_mu[j,...] DxN matrix
			Sj = matmul(X_mu[j,...]*Hj,X_mu[j,...].T)
			psi_new += np.diag( Sj - matmul(W_new[j,...], matmul(Ez[j,...],X_mu[j,...].T) ) ).reshape((1,-1))
		psi_new /= N

		# Update all parameters for next itr
		pi = pi_new  # (M,) array
		mu = mu_new  # MxDx1 matrix
		# print('W_new.shape={}'.format(W_new.shape))
		W = W_new  # MxDxK matrix
		psi = psi_new  # 1xD matrix

	return pi, mu.reshape(M,D), W, psi[0]