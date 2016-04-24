import numpy as np
#import pyGM as gm

#Part A

# Load the data points D, and the station locations (lat/lon)
D = np.genfromtxt('data/data.txt',delimiter=None)
loc = np.genfromtxt('data/locations.txt',delimiter=None)
m,n = D.shape # m = 2760 data points, n=30 dimensional
# D[i,j] = 1 if station j observed rainfall on day i
print m,n
print loc.shape

#Part B

#this find phat for each variable
posXcount = np.sum(D,axis=0)
probXhat1 = np.divide(posXcount,m)
probXhat0 = 1-probXhat1

#this find phat(x_j,x_k) for each pair of variables
probXjk00 = np.zeros((n,n))
probXjk01 = np.zeros((n,n))
probXjk10 = np.zeros((n,n))
probXjk11 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        for k in range(m):
            if D[k, i] == 0:
                if D[k, j] == 0:
                    probXjk00[i, j] += 1
                if D[k, j] == 1:
                    probXjk01[i, j] += 1
            if D[k, i] == 1:
                if D[k, j] == 0:
                    probXjk10[i, j] += 1
                if D[k, j] == 1:
                    probXjk11[i, j] += 1
probXjk00 = np.divide(probXjk00,m)
probXjk01 = np.divide(probXjk01,m)
probXjk10 = np.divide(probXjk10,m)
probXjk11 = np.divide(probXjk11,m)

mutualInfo = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        denomTerm00 = probXhat0[i]*probXhat0[j]
        denomTerm01 = probXhat0[i] * probXhat1[j]
        denomTerm10 = probXhat1[i] * probXhat0[j]
        denomTerm11 = probXhat1[i] * probXhat1[j]

        term00 = probXjk00[i,j]*np.log(probXjk00[i,j]/denomTerm00)
        term01 = probXjk01[i, j] * np.log(probXjk01[i, j] / denomTerm01)
        term10 = probXjk10[i, j] * np.log(probXjk10[i, j] / denomTerm10)
        term11 = probXjk11[i, j] * np.log(probXjk11[i, j] / denomTerm11)

        mutualInfo[i,j]=term00+term01+term10+term11

print mutualInfo

