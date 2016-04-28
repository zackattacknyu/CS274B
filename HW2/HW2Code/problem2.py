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
# for i in range(len(listVertices)):
#     print listVertices[i],adjList[i]
#
# for jj in range(len(listVertices)):
#     curJind = listVertices[jj]
#     for kk in adjList[jj]:
#         print probXjGivenK[curJind,kk,:,:]

#Part C

#makes the variables
gmNodes = [gm.Var(i,2) for i in range(n)]
#print gmNodes

#makes the factors given the loopy model graph we have
gmFactors = []
for jj in range(len(listVertices)):
    curJind = listVertices[jj]
    for kk in adjList[jj]:
        gmFactors.append(gm.Factor([gmNodes[curJind],gmNodes[kk]],1.0))

#fills the table with the empiricial probabilities we have calculated
curListInd = 0
for jj in range(len(listVertices)):
    curJind = listVertices[jj]
    for kk in adjList[jj]:
        #gmFactors[curListInd].table = probXjk[curJind,kk,:,:]
        inputFactor = np.matrix(np.ones((2,2)))
        inputFactor = np.multiply(inputFactor,0.25)
        gmFactors[curListInd].table = inputFactor
        curListInd += 1

#for ii in range(len(gmFactors)):
#    print gmFactors[ii]

listInd = 0
sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate
for jj in range(len(listVertices)):
    curJind = listVertices[jj]
    for kk in adjList[jj]:
        print jj,kk

        if listInd>1:
            print gmFactors[listInd-1].table

        #does variable elimination to get p_ij value
        currentFactors = copy.deepcopy(gmFactors)
        curModel = gm.GraphModel(currentFactors)
        pri = [1.0 for Xi in currentFactors]
        pri[curJind], pri[kk] = 2.0, 2.0
        order = gm.eliminationOrder(curModel,orderMethod='minwidth',priority=pri)[0]
        curModel.eliminate(order[:-2], sumElim)  # eliminate all but last two
        curP = curModel.joint()
        curLnZ = np.log(curP.sum())
        print 'lnZ: ', curLnZ, '\n'
        curP /= curP.sum()

        curLog = 0
        for ptNum in range(m):
            curLog += curModel.logValue(D[ptNum,:])
        print curLog/m
        #update the factor
        currentFij = gmFactors[listInd].table
        probRatio = np.matrix(np.divide(probXjk[curJind,kk,:,:],curP.table))
        newFij = np.multiply(currentFij,probRatio)
        gmFactors[listInd].table = newFij
        listInd+=1

# factors = [ gm.Factor([X[0],X[1]],1.0) , gm.Factor([X[0],X[2]],1.0) , gm.Factor([X[1],X[2]],1.0),
#             gm.Factor([X[1],X[3]],1.0) , gm.Factor([X[2],X[3]],1.0) , gm.Factor([X[3],X[4]],1.0),
#             gm.Factor([X[2],X[4]],1.0) , gm.Factor([X[4],X[5]],1.0) , gm.Factor([X[2],X[5]],1.0)]
#
# for i in range(len(factors)):               # fill the tables with random values
#     factors[i].table = np.random.rand(3,3)
#
# # Perform variable elimination
# model_ve = gm.GraphModel(factors) # make a new model (will be modified by VE)
#
# pri = [1.0 for Xi in X]
# pri[1], pri[3] = 2.0,2.0          # eliminate X1 and X3 last
# order = gm.eliminationOrder(model_ve, orderMethod='minfill', priority=pri)[0]
# print pri
# print order,'\n'
#
# sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate
# model_ve.eliminate(order[:-2], sumElim)  # eliminate all but last two
#
# p13 = model_ve.joint()
# lnZ = np.log(p13.sum())   # can get the (log) partition function as well
# print 'lnZ: ',lnZ,'\n'
# p13 /= p13.sum()
# print p13, '\n',p13.table
#
# print len(factors)
