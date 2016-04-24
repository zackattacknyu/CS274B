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

plt.hold(True)
plt.plot(loc[:,1],loc[:,0],'ro')
for i in range(mm):
    for j in range(i+1,mm):
        node0 = loc[i,:]
        node1 = loc[j,:]
        xx = [node0[1],node1[1]];
        yy = [node0[0],node1[0]]
        if testadj[i,j]>0:
            plt.plot(xx,yy,'b-')
plt.show()