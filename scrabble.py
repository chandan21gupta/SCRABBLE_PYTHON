import scipy.io
import numpy as np
from scipy.linalg import svd
import ctypes
from cDescent_1 import cDescent
import matplotlib.pyplot as plt
#so_file = '/home/upriverbasil/Downloads/SCRABBLE_PYTHON-master/cDescent.so'
#cfactorial = ctypes.CDLL(so_file)
x = []
def scrabble_optimization(data_path = './demo_data.mat', parameters = [1, 1e-6, 1e-4], nIter = 20, nIter_inner = 20, error_inner_threshold = 1e-4, error_outer_threshold = 1e-4):
	
	data = scipy.io.loadmat('demo_data.mat')
	Y = data['data_sc'].transpose()
	projection = Y > 0

	alpha = parameters[0]
	beta = parameters[1]
	gamma = parameters[2]

	if 'data_bulk' not in data:
		beta = 0
		Z = np.ones((1, Y.shape[1]))
		print("No bulk RNAseq data provided. Continuing without it with parameter beta = 0")

	else:
		Z = data['data_bulk']*Y.shape[1]

	n = Y.shape[0]
	D = np.ones((1, n))
	A = beta*(np.dot(np.transpose(D), D)) + gamma*(np.identity(n))
	print(D.shape,Z.shape,Y.shape)
	B = beta*np.dot(D.T, Z.T) + Y

	X = Y
	newX = X
	newY = Y
	Lambda = np.zeros(shape = Y.shape)
	#gamma =gamma.astype(np.double)
	Y = Y.astype(np.double)
	B = B.astype(np.double)
	A = A.astype(np.double)
	gamma = float(gamma)
	projection = projection.astype(np.double)
	newX = newX.astype(np.double)
	m1,n1 = X.shape[0], X.shape[1]
	# cfactorial.cDescent.argtypes = [ctypes.c float, ,c_double_p,c_double_p,c_double_p,c_double_p,c_double_p,c_double_p ]
	print("SCRABBLE begins...")
	k = 0
	error = 1
	origX=X
	while k < nIter and error > error_outer_threshold:
		X = newX
		Y = newY
		l = 1
		error_inner = 1
		X1 = newX
		while error_inner > error_inner_threshold and l < nIter_inner:
			m1 = Y.shape[0]
			n1 = Y.shape[1]
			x = np.zeros((m1,n1))
			newX = cDescent(gamma, Y, B, Lambda, A, projection, newX, x, n1, m1)

			l = l+1
			error_inner = np.linalg.norm(np.log10(X1+1)-np.log10(newX+1), ord = 'fro')/(m1*n1)
			X1 = newX
			#print(error_inner,error_inner_threshold)
			print('The %d-th INNNER iteration and the error is %1.4e\n'%(l,error_inner))
		S = (newX + Lambda)/gamma
		tau = alpha/gamma
		#u, s, v = svt(S, 'lambda', tau)
		u, s, v = svd(S,full_matrices=False)

		#newY = u*np.diag(s-tau).T*np.transpose(v)
		#print(u.shape,(s-tau).shape,v.shape,s.shape)
		newY = np.dot(u,np.diag(s-tau)*np.transpose(v))
		error = np.linalg.norm(np.log10(X+1)-np.log10(newX+1), ord = 'fro')/(m1*n1)
		if k == 0:
			error = 1
		k = k+1
		Lambda = Lambda+gamma*(newX-newY)
		print('The %d-th iteration and the error is %1.4e\n'%(k,error))
	fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
	ax1.imshow(np.log10(origX+1))
	ax1.set_title("Orignal Data")
	ax2.imshow(np.log10(newX+1))
	ax2.set_title("Imputed Data")
	ax3.imshow(np.log10(projection+1))
	ax3.set_title("Dropout")
	fig.show()
	plt.show()
	print('SCRABBLE finished the imputation!')

	return np.transpose(newX)

scrabble_optimization()










