import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
# create problem data 
N = 100; 

# create an increasing input signal
xtrue = np.zeros((N,1))
xtrue[1:40] = 0.1
xtrue[50] = 2
xtrue[70:80] = 0.15
xtrue[80] = 1
xtrue = np.cumsum(xtrue)

# pass the increasing input through a moving-average filter 
# and add Gaussian noise
h = np.array([1, -0.85 ,0.7 ,-0.3])
k = h.shape[0]
yhat = np.convolve(h,xtrue)
y = yhat[:-3] + np.random.randn(N)
x = cp.Variable((100,),nonneg = True)
z = y[:,None] - cp.conv(h,x)[:-3]
objective = cp.Minimize(cp.sum_squares(z))
constraints = [cp.diff(x) >= 0]
prob=cp.Problem(objective,constraints=constraints)
prob.solve()

#plot
t = list(range(0,xtrue.size))
plt.plot(t,list(xtrue), color='red',label='x_true')
plt.plot(t,list(x.value), color='blue',label='x_hat')
plt.legend(loc="upper left")
plt.savefig('prob_66.png')
plt.show()