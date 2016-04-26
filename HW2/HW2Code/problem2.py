import numpy as np
import matplotlib.pyplot as plt

edges = np.genfromtxt('data/edges.txt')
loc = np.genfromtxt('data/locations.txt',delimiter=None)
D = np.genfromtxt('data/data.txt',delimiter=None)
m,n = D.shape # m = 2760 data points, n=30 dimensional

nEdges,cc = edges.shape
nNodes,cd = loc.shape

adjMatrix = np.zeros((nNodes,nNodes))
for curI in range(nEdges):
    curNode0 = edges[curI, 0]
    curNode1 = edges[curI, 1]
    adjMatrix[curNode0,curNode1] = 1

print loc.shape

#Part A
showAplot = False
if showAplot:
    plt.hold(True)
    plt.plot(loc[:,1],loc[:,0],'ro')
    for i in range(nNodes):
        for j in range(i+1,nNodes):
            node0 = loc[i,:]
            node1 = loc[j,:]
            xx = [node0[1],node1[1]];
            yy = [node0[0],node1[0]]
            if adjMatrix[i,j]>0:
                plt.plot(xx,yy,'b-')
    plt.title('Weather Station Locations with Loopy Model')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

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

#find phat(x_j|x_k)
probXjGivenK = np.zeros((n,n,2,2))
for i in range(n):
    for j in range(n):
        for jval in range(2):
            for kval in range(2):
                probXjGivenK[i,j,jval,kval] = probXjk[i,j,jval,kval]/probXj[j,kval]

def getAdjList(adjMatrix):
    mm,xx = adjMatrix.shape
    visited = np.zeros((mm))
    adjList = []
    listVertices = []
    for i in range(mm):
        visited[i] = 1
        adjVertices = np.where(adjMatrix[i,:]>0)
        verticesAdd = []
        for j in adjVertices[0]:
            if visited[j] <= 0:
                verticesAdd.append(j)
        if len(verticesAdd) > 0:
            adjList.append(verticesAdd)
            listVertices.append(i)
    return listVertices,adjList

listVertices,adjList = getAdjList(adjMatrix)
for i in range(len(listVertices)):
    print listVertices[i],adjList[i]

for jj in range(len(listVertices)):
    curJind = listVertices[jj]
    for kk in adjList[jj]:
        print probXjGivenK[curJind,kk,:,:]

