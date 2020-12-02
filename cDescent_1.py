import numpy as np
import copy
def cDescent(gamma, y, b, lamda, a, zones, newX, x, n1, m1):
	x=copy.deepcopy(newX)
	for i in range(0,m1):
		for j in range(0,n1):
			temp = 0
			for k in range(0,m1):
				temp+=a[i][k]*x[k][j]
			temp = (gamma*y[i][j]+b[i][j]-lamda[i][j]-temp+a[i][i]*x[i][j])/(a[i][i]+zones[i][j])
			x[i][j] = max(temp,0)
	return x
	#return np.random.rand(m1,n1)