import numpy as np
import matplotlib.pyplot as plt
import pyGM as gm
import networkx as nx
import copy

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

#print loc.shape

#Part A

showNxPlot = False
if showNxPlot:
    graph2 = nx.Graph()
    graph2.add_nodes_from(range(n))
    for edgeI in edges:
        graph2.add_edge(edgeI[0], edgeI[1])
    loc2 = np.zeros(loc.shape)
    loc2[:,0] = loc[:,1]
    loc2[:,1] = loc[:,0]
    nx.draw_networkx(graph2, node_color='c', pos=loc2)
    plt.title('Weather Station Locations with Loopy Model')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

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

edgeMarginals = np.zeros((nEdges,2,2))
for ee in range(nEdges):
    node0 = edges[ee,0]
    node1 = edges[ee,1]
    edgeMarginals[ee,:,:] = probXjk[node0,node1,:,:]
print edgeMarginals[0:5]

#find phat(x_j|x_k)
# probXjGivenK = np.zeros((n,n,2,2))
# for i in range(n):
#     for j in range(n):
#         for jval in range(2):
#             for kval in range(2):
#                 probXjGivenK[i,j,jval,kval] = probXjk[i,j,jval,kval]/probXj[j,kval]

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

#Part C

#makes the variables

#tests code on example in slides
#Gave sample p12 and p13 result, so the var elim is working correctly
# n=3
# adjMatrix = np.ones((3,3))
# listVertices,adjList = getAdjList(adjMatrix)
# gmNodes = [gm.Var(i,3) for i in range(n)]
# probNodes = [gm.Var(i,3) for i in range(n)]
# probXjk = np.zeros((3,3,3,3))
# probXjk[0,1,:,:] = [[0.249,0.002,0.311],[0.017,0.024,0.015],[0.029,0.348,0.000]]
# probXjk[0,2,:,:] = [[0.136,0.118,0.309],[0.003,0.029,0.025],[0.0348,0.014,0.015]]

#initializes the factors given the loopy model graph we have
gmNodes = [gm.Var(i,2) for i in range(nEdges)]
probNodes = [gm.Var(i,2) for i in range(nEdges)]
gmFactors = []
#probFactors = []
for ee in range(nEdges):
    jj = int(edges[ee,0])
    kk = int(edges[ee,1])
    gmFactors.append(gm.Factor([gmNodes[jj], gmNodes[kk]], 1.0))
    #probFactors.append(gm.Factor([probNodes[jj], probNodes[kk]], 1.0))

    #fills the table with probabilities
    inputFactor = np.matrix(np.ones((2, 2)))

    #inputFactor = np.multiply(inputFactor,0.25)
    gmFactors[ee].table = inputFactor
    #probFactors[ee].table = probXjk[jj, kk, :, :]


sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate

numIter=15
totalEnt = numIter*nEdges
logLikeIter = np.zeros(numIter)
logLikeAll = np.zeros(totalEnt)
arrInd = 0
for iterI in range(numIter):
    print 'Now computing Iteration: ',iterI
    for ee in range(nEdges):
        #print 'Now processing edge: ',ee
        #print jj,kk



        #current edge
        jj = int(edges[ee, 0])
        kk = int(edges[ee, 1])

        #does variable elimination to get p_jk value
        currentFactors = copy.deepcopy(gmFactors)
        curModel = gm.GraphModel(currentFactors)
        pri = [1.0 for Xi in currentFactors]
        pri[jj], pri[kk] = 2.0, 2.0
        order = gm.eliminationOrder(curModel,orderMethod='minfill',priority=pri)[0]
        curModel.eliminate(order[:-2], sumElim)  # eliminate all but last two
        curP = curModel.joint()
        curLnZ = np.log(curP.sum())
        curP /= curP.sum()
        #print curP.table

        #update the current f_jk value
        #currentFij = gmFactors[ee].table
        #probRatio = np.matrix(np.divide(probXjk[jj,kk,:,:],curP.table))
        #newFij = np.multiply(currentFij,probRatio)
        #gmFactors[ee].table = newFij

        newFij = gmFactors[ee].table*probXjk[jj,kk,:,:]/curP.table
        gmFactors[ee].table = newFij

        #update the probabilistic model
        #newFijNorm = newFij/newFij.sum()
        #probFactors[ee].table = newFijNorm

    # computes the likelihood of the current model
    curLog = 0
    probModel = gm.GraphModel(gmFactors)
    for ptNum in range(m):
        curLog += probModel.logValue(D[ptNum, :])#-curLnZ
    curLog = curLog / m - curLnZ
    logLikeIter[iterI] = curLog
    print 'logLike: ',curLog
    print 'lnZ: ', curLnZ


plt.plot(logLikeIter)
plt.xlabel('Number of Complete Iterations Done')
plt.ylabel('Log Likelihood Of Model')
plt.show()

print logLikeIter