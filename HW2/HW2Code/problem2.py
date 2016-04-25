import numpy as np
import matplotlib.pyplot as plt

edges = np.genfromtxt('data/edges.txt')
loc = np.genfromtxt('data/locations.txt',delimiter=None)

nEdges,cc = edges.shape
nNodes,cd = loc.shape

adjMatrix = np.zeros((nNodes,nNodes))
for curI in range(nEdges):
    curNode0 = edges[curI, 0]
    curNode1 = edges[curI, 1]
    adjMatrix[curNode0,curNode1] = 1

print loc.shape

#Part A

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