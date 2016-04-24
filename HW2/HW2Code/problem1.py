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
probXj = np.zeros((n,2))
probXj[:, 1] = np.divide(posXcount,m)
probXj[:, 0] = 1-probXj[:,1]

#this find phat(x_j,x_k) for each pair of variables
# next dimension is probability of j, then prob of k
probXjk = np.zeros((n,n,2,2))
for i in range(n):
    for j in range(n):
        for k in range(m):
            probXjk[i, j, D[k, i], D[k, j]] += 1
probXjk = np.divide(probXjk,m)

#print probXjk

mutualInfo = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        curTerm = 0
        for k1 in range(2):
            for k2 in range(2):
                denomTerm = probXj[i,k1]*probXj[i,k2]
                jointTerm = probXjk[i, j, k1, k2]
                if denomTerm>0 and jointTerm>0:
                    curTerm += jointTerm*np.log(jointTerm/denomTerm)
        mutualInfo[i, j] = curTerm

#each entry should be close to zero, which does occur
#print mutualInfo-np.transpose(mutualInfo)

#for verification, calculate entropy
entropy = np.zeros(n)
for i in range(n):
    enTerm = 0
    for k in range(2):
        curProbTerm = probXj[i,k]
        enTerm += curProbTerm*np.log(curProbTerm)
    entropy[i] = -enTerm
#print entropy

#look at diagnoal of mutual info matrix for entropy
mutualInfoEntropy = np.zeros(n)
for i in range(n):
    mutualInfoEntropy[i] = mutualInfo[i,i]
#print mutualInfoEntropy

#Part C
#I will add max weight edges

edges = np.zeros(n*(n-1)/2)
edgeNodeID = np.zeros((n*(n-1)/2,2))
edgeInd = 0
for i in range(n):
    for j in range(i+1,n):
        edges[edgeInd] = mutualInfo[i, j]
        edgeNodeID[edgeInd,:] = [i, j]
        edgeInd+=1
#print edges

sortedEdges = np.sort(edges)[::-1]
sortedEdgeID = np.argsort(edges)[::-1]
#print sortedEdges

adjMatrix = np.zeros((n,n))
for curI in range(len(sortedEdges)):
    curEdgeID = sortedEdgeID[curI]
    curNode0 = edgeNodeID[curEdgeID, 0]
    curNode1 = edgeNodeID[curEdgeID, 1]
    #check if path of any length between node 0 and 1
    #add to adj matrix if not



