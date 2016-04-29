import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
                denomTerm = probXj[i,k1]*probXj[j,k2]
                jointTerm = probXjk[i, j, k1, k2]
                if denomTerm>0 and jointTerm>0:
                    curTerm += jointTerm*np.log(jointTerm/denomTerm)
        mutualInfo[i, j] = curTerm
print mutualInfo[0:5,0:5]
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

def nodesHavePath(adjMatrix,node0,node1):

    n,m = adjMatrix.shape
    adjMatrix = adjMatrix + np.transpose(adjMatrix)
    adjPower = np.copy(adjMatrix)
    adjTotal = np.copy(adjMatrix)
    for i in range(n):
        adjPower = np.dot(adjMatrix, adjPower)
        adjTotal += adjPower

    return (adjTotal[node0,node1]>0)


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
    if not nodesHavePath(adjMatrix,curNode0,curNode1):
        adjMatrix[curNode0,curNode1]=1

#print adjMatrix

showGraph = False #change to true if I want to display it
if showGraph:
    plt.hold(True)
    plt.plot(loc[:,1],loc[:,0],'ro')
    for i in range(n):
        for j in range(i+1,n):
            node0 = loc[i,:]
            node1 = loc[j,:]
            xx = [node0[1],node1[1]];
            yy = [node0[0],node1[0]]
            if adjMatrix[i,j]>0:
                plt.plot(xx,yy,'b-')
    plt.title('Weather Station Locations with Chow-Liu Tree')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

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

showNxPlot = True
if showNxPlot:
    graph2 = nx.Graph()
    graph2.add_nodes_from(range(n))
    for i in range(len(listVertices)):
        for j in adjList[i]:
            graph2.add_edge(listVertices[i],j)
    loc2 = np.zeros(loc.shape)
    loc2[:, 0] = loc[:, 1]
    loc2[:, 1] = loc[:, 0]
    nx.draw_networkx(graph2, node_color='c', pos=loc2)
    plt.title('Weather Station Locations with Chow-Liu Tree')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

#Part D
# Calculate the likelihood
# loglike = 0
# for ii in range(m):
#     sampleVals = D[ii,:]
#     for jj in range(len(listVertices)):
#         #This calculates the p(x_j) term in log-likelihood
#         curJind = listVertices[jj]
#         curXjVal = sampleVals[curJind]
#         curProbXj = probXj[curJind,curXjVal]
#         loglike += np.log(curProbXj)
#         for kk in adjList[jj]:
#             curXkval = sampleVals[kk]
#             jointProb = probXjk[curJind,kk,curXjVal,curXkval]
#             marginalProb = jointProb/curProbXj
#             loglike += np.log(marginalProb)
# print loglike/m

#Calculate likelihood using Slide 6 method of summing empiricial entropy and mutual information
entropyPart = entropy.sum()
#print entropy
#print entropyPart

mutualInfoPart = 0
for jj in range(len(listVertices)):
    curJind = listVertices[jj]
    for kk in adjList[jj]:
        mutualInfoPart += mutualInfo[curJind,kk]
#print mutualInfoPart

loglike = mutualInfoPart - entropyPart

print 'Log Likelihood:'
print loglike