import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
m=100
k=40 # max # permuted measurements
n=20
A=10*np.random.randn(m,n)
x_true=np.random.randn(n) # true x value
y_true = A.dot(x_true) + np.random.randn(m)
# build permuted indices
perm_idxs=np.random.permutation(m)
perm_idxs=np.sort(perm_idxs[:k])
temp_perm=np.random.permutation(k)
new_pos=np.zeros(k)
for i in range(k):
    new_pos[i] = perm_idxs[temp_perm[i]]
new_pos = new_pos.astype(int)
# true permutation matrix
P=np.identity(m)

P[perm_idxs]=P[new_pos,:]
true_perm=[]
for i in range(k):
    if perm_idxs[i] != new_pos[i]:
        true_perm = np.append(true_perm, perm_idxs[i])
y=P.dot(y_true)
new_pos = None
# naive estimator (P=I)
x_naive = np.linalg.lstsq(A,y)[0]
# robust estimator
x_hub = cp.Variable(n)
obj = cp.sum(cp.huber(A@x_hub-y))  #TODO dimention bugs
cp.Problem(cp.Minimize(obj)).solve()
plt.figure(1)
plt.plot(np.arange(m), np.abs(A.dot(x_hub.value)-y), '.')
plt.ylabel('residual')
plt.xlabel('idx')
plt.savefig('prob_152.png')

# remove k largest residuals
cand_idxs = np.zeros(m)
cand_idxs[:] = np.flip(np.argsort(np.abs(A.dot(x_hub.value)-y)))
cand_idxs = np.sort(cand_idxs[:k])
cand_idxs = cand_idxs.astype(int)
keep_idxs = np.zeros(m)
keep_idxs[:] = np.argsort(np.abs(A.dot(x_hub.value)-y).T)
keep_idxs = np.sort(keep_idxs[:(m-k)])
keep_idxs = keep_idxs.astype(int)
# print(np.shape(A))
# print(np.shape(y))

A_hat = A[keep_idxs,:]
# print(np.shape(A_hat))
y_hat = y[keep_idxs]
# print(np.shape(y_hat))

# ls estimate with candidate idxs removed
x_ls = np.linalg.lstsq(A_hat,y_hat)[0]
# match predicted outputs with measurements
b = np.zeros(k)
c = np.zeros(k)
b[:] = np.argsort(A[cand_idxs,:].dot(x_ls).T)
b = b.astype(int)
c[:] = np.argsort(y[cand_idxs].T)
c = c.astype(int)
# reorder A matrix
cand_perms = np.zeros(len(cand_idxs))
cand_perms[:]=cand_idxs[:]
cand_perms[b]=cand_perms[c]
cand_perms = cand_perms.astype(int)
A[cand_perms,:]=A[cand_idxs,:]
x_final = np.linalg.lstsq(A,y)[0]
# final estimate of permuted indices
perm_estimate = []
for i in range(k):
    if cand_perms[i] != cand_idxs[i]:
        perm_estimate = np.append(perm_estimate, cand_idxs[i])
naive_error = np.linalg.norm(x_naive-x_true)
final_error = np.linalg.norm(x_final-x_true)