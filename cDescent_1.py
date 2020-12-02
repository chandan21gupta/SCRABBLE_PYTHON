def cDescent(gamma, y, b, lamda, a, zones, newX, x, n1, m1):
	for i in range(0,m1):
		for j in range(0,n1):
			index = i+m1*j
			print(i,j)
			x[index] = newX[index]
	for i in range(0,m1):
		for j in range(0,n1):
			temp = 0
			for k in range(0,m1):
				temp+=a[i+m1*k]*x[k+m1*j]
			index = i+m1*j
			temp = (gamma*y[index]+b[index]-lamda[index]-temp+a[i+m1*i]*x[index])/(a[i+m1*i]+zones[index])
			x[index] = max(temp,0)