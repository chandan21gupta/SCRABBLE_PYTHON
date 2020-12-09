import scipy.io
import numpy as np
from scipy.linalg import svd
import ctypes
from cDescent_1 import cDescent
import matplotlib.pyplot as plt

x = []
def scrabble_optimization(data_path = './demo_data.mat', parameters = [1, 1e-6, 1e-4], nIter = 20, nIter_inner = 20, error_inner_threshold = 1e-4, error_outer_threshold = 1e-4):
	'''
	Scrabble Function to impute input data matrix.

	Parameters
	-----------

	data_path = path to .mat file

	paramters = alpha -> weight for the rank of the imputed data matrix.
				beta -> weight for the agreement between the aggregated scRNA-seq and bulk RNA-seq data.
				gamma -> penalty parameter.
	
	nIter = Number of iterations to run the optimization for. (Default 20)

	nIter_inner = Number of iterations to run the inner optimization for. (Default 20)

	error_inner_threshold = Error threshold for inner optimization loop. (Default 1e-4)
	
	error_outer_threshold = Error threshold for outer optimization loop. (Default 1e-4)

	----------
	Returns
	----------

	newX = Imputed Data Matrix
	'''
	data = scipy.io.loadmat('demo_data.mat')
	original_data_without_droputs = data['data_true'].transpose()
	Y = data['data_sc'].transpose()
	projection = Y > 0
	projection2 = Y <= 0

	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	#Check Data
	if 'data_bulk' not in data:
		beta = 0
		Z = np.ones((1, Y.shape[1]))
		print("No bulk RNAseq data provided. Continuing without it with parameter beta = 0")

	else:
		Z = (data['data_bulk'].transpose())*Y.shape[1]

	#Initialise Variables
	n = Y.shape[0]
	D = np.ones((1, n))
	A = beta*(np.dot(np.transpose(D), D)) + gamma*(np.identity(n))
	B = np.dot(np.dot(beta,D.T),Z) + Y

	X = np.copy(Y)
	newX = np.copy(X)
	newY = np.copy(Y)
	Lambda = np.zeros(shape = Y.shape)

	Y = Y.astype(np.double)
	B = B.astype(np.double)
	A = A.astype(np.double)
	gamma = float(gamma)
	projection = projection.astype(np.double)
	newX = newX.astype(np.double)
	m1,n1 = X.shape[0], X.shape[1]

	print("SCRABBLE begins...")
	k = 0
	error = 1
	while k < nIter and error > error_outer_threshold:
		X = np.copy(newX)
		Y = np.copy(newY)
		l = 1
		error_inner = 1
		X1 = np.copy(newX)
		while error_inner > error_inner_threshold and l < nIter_inner:
			x = np.zeros((m1,n1))
			#calculate cDesent
			newX = cDescent(gamma, Y, B, Lambda, A, projection, newX, x, n1, m1)

			l = l+1
			error_inner = np.linalg.norm(np.log10(X1+1)-np.log10(newX+1), ord = 'fro')/(m1*n1)
			X1 = np.copy(newX)
			#print(error_inner,error_inner_threshold)
			print('The %d-th INNNER iteration and the error is %1.4e\n'%(l,error_inner))
		S = newX + Lambda/gamma
		tau = alpha/gamma

		#calculate SVT
		u, s, v = svd(S,full_matrices=False)
		newY = np.dot(np.dot(u,np.diag(s-tau)),np.transpose(v))

		error = np.linalg.norm(np.log10(X+1)-np.log10(newX+1), ord = 'fro')/(m1*n1)
		if k == 0:
			error = 1
		k = k+1
		Lambda = Lambda+gamma*(newX-newY)
		print('The %d-th iteration and the error is %1.4e\n'%(k,error))

	#Plot the Results
	fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
	ax1.imshow(np.log10(original_data_without_droputs+1))
	ax1.set_title("Orignal Data") 
	ax2.imshow(np.log10(newY+1))
	ax2.set_title("Imputed Data")
	ax3.imshow(np.log10(Y+1))
	ax3.set_title("Dropout")
	fig.show()
	plt.show()
	print('SCRABBLE finished the imputation!')

	return np.transpose(newX)

X=scrabble_optimization()










