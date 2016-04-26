import numpy as np
import matplotlib.pyplot as plt

adjMatrix = np.zeros((5,5))
adjMatrix[0,1] = 1
adjMatrix[1,2] = 1
adjMatrix[3,4] = 1

def nodesHavePath(adjMatrix,node0,node1):

    n,m = adjMatrix.shape
    adjMatrix = adjMatrix + np.transpose(adjMatrix)
    adjPower = np.copy(adjMatrix)
    adjTotal = np.copy(adjMatrix)
    for i in range(n):
        adjPower = np.dot(adjMatrix, adjPower)
        adjTotal += adjPower

    return (adjTotal[node0,node1]>0)

print nodesHavePath(adjMatrix,0,1)
print nodesHavePath(adjMatrix,0,3)
print nodesHavePath(adjMatrix,2,3)
print nodesHavePath(adjMatrix,3,4)

loc = np.genfromtxt('data/locations.txt',delimiter=None)

print loc.shape

mm,nn = loc.shape;

testadj = np.zeros((mm,mm))
testadj[1,2] = 1
testadj[1,11] = 1
testadj[3,4] = 1
testadj[4,5] = 1
testadj[4,12] = 1
testadj[6,7] = 1
testadj[3,8] = 1
testadj[5,10] = 1
#
# plt.hold(True)
# plt.plot(loc[:,1],loc[:,0],'ro')
# for i in range(mm):
#     for j in range(i+1,mm):
#         node0 = loc[i,:]
#         node1 = loc[j,:]
#         xx = [node0[1],node1[1]];
#         yy = [node0[0],node1[0]]
#         if testadj[i,j]>0:
#             plt.plot(xx,yy,'b-')
# plt.show()

# arr1 = np.array([1,2,3])
# arr2 = np.array([4,5,6,7])
# test1 = [arr1,arr2]
# arr3 = np.array([45,56])
# test1.append(arr3)
# print test1[0]
# print test1[2]
# print test1[1]

testadj = testadj + np.transpose(testadj)

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

listVertices,adjList = getAdjList(testadj)
for i in range(len(listVertices)):
    print listVertices[i],adjList[i]


