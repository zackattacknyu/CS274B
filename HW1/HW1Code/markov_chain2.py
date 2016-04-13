import numpy as np
from os import walk
import numpy.matlib


mypath = 'proteins/'  # use path to data files
_, _, filenames = next(walk(mypath), (None, None, []))

mSeq = len(filenames)        # read in each sequence
o,x = [],[]
for i in range(mSeq):
    f = open( mypath + filenames[i] , 'r')
    o.append( f.readline()[:-1] )  # strip trailing '\n'
    x.append( f.readline()[:-1] )
    f.close()

xvals, ovals = set(),set()  # extract the symbols used in x and o
for i in range(mSeq):
    xvals |= set(x[i])
    ovals |= set(o[i])
xvals = list( np.sort( list(xvals) ) )
ovals = list( np.sort( list(ovals) ) )
dx,do = len(xvals),len(ovals)

for i in range(mSeq):       # and convert to numeric indices
    x[i] = np.array([xvals.index(s) for s in x[i]])
    o[i] = np.array([ovals.index(s) for s in o[i]])

#Part A Code
p0vals = np.zeros(len(xvals))
for i in range(mSeq):
    curSeq = x[i]
    x0val = curSeq[0]
    p0vals[x0val] += 1
p0vals = np.divide(p0vals,mSeq)
print 'p(x_0) is as follows:'
print np.transpose(np.matrix(p0vals))


#Part B Code
Tmatrix = np.zeros((len(xvals),len(xvals)))
for i in range(mSeq):
    curSeq = x[i]
    for j in range(1,len(curSeq)):
        xPrev = curSeq[j-1]
        xCurrent = curSeq[j]
        Tmatrix[xPrev,xCurrent] += 1
Tsum = np.matrix(np.sum(Tmatrix,0))
Tsum = np.transpose(Tsum)
#TsumTiled = np.tile(Tsum,(1,8))
#print Tsum.shape
TsumTiled = np.matlib.repmat(Tsum,1,8)
Tmatrix = np.divide(Tmatrix,TsumTiled)
#print
#print np.sum(Tmatrix,1) #should be column of all ones


print
print 'Transition Matrix (first 5 states) is as follows:'
print Tmatrix[0:5,0:5]

epsilon = 1e-8 #whether or not we have stability

#toy example
toyT = np.matrix([[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.6,0.1]])
toyX0 = np.matrix([0.1,0.2,0.7])
curToyX = toyX0*toyT
print
for i in range(500):
    prevToyX = np.copy(curToyX)
    curToyX = curToyX*toyT
    diffX = np.abs(np.subtract(curToyX,prevToyX))
    if np.sum(diffX)<epsilon:
        print 'Stable Toy at: ' + str(i)
        print curToyX
        break
#print np.sum(curToyX)

#Part B, getting the stationary distribution
Tmatrix2 = np.matrix(Tmatrix)
x0array = np.matrix(p0vals)
x1array = x0array*Tmatrix2
print
#print Tmatrix2.shape
#print x0array.shape
#print x1array
#print np.sum(x1array)

curX = x1array
for i in range(500):
    prevX = np.copy(curX)
    curX = curX*Tmatrix2
    diffX = np.abs(np.subtract(prevX,curX))
    if np.sum(diffX)<epsilon:
        print 'Stable State at: ' + str(i)
        break
print 'Stationary Distribution:'
print np.transpose(np.matrix(curX))
#print np.sum(curX)

#Part C Code
Omatrix = np.zeros((len(xvals),len(ovals)))
for i in range(mSeq):
    curSeq = x[i]
    curObs = o[i]
    for j in range(len(curSeq)):
        xt = curSeq[j]
        ot = curObs[j]
        Omatrix[xt,ot] += 1
Osum = np.matrix(np.sum(Omatrix,0))
OsumTiled = np.tile(Osum,(8,1))
Omatrix = np.divide(Omatrix,OsumTiled)
#print np.sum(Omatrix,0) #shows each column sums to 1
print
print 'Emission Probability Matrix (first 5 states) is as follows:'
print Omatrix[0:4,0:4]



# function markovMarginals(x,o,p0,Tr,Ob):
#     '''Compute p(o) and the marginal probabilities p(x_t|o) for a Markov model
#        defined by P[xt=j|xt-1=i] = Tr(i,j) and P[ot=k|xt=i] = Ob(i,k) as numpy matrices'''
#     dx,do = Ob.shape()   # if a numpy matrix
#     L = len(o)
#     f = np.zeros((L,dx))
#     r = np.zeros((L,dx))
#     p = np.zeros((L,dx))
#
#     f[0,:] = ...   # compute initial forward message
#     log_pO = ...   # update probability of sequence so far
#     f[0,:] /= f[0,:].sum()  # normalize (to match definition of f)
#
#     for t in range(1,L):    # compute forward messages
#         f[t,:] = ...
#         log_pO += ...
#         f[t,:] /= f[t,:].sum()
#
#     r[L,:] = 1.0  # initialize reverse messages
#     p[L,:] = ...  # and marginals
#
#     for t in range(L-1,-1,-1):
#         r[t,:] = ...
#         r[t,:] /= r[t,:].sum()
#         p[t,:] = ...
#         p[t,:] /= p[t,:].sum()
#
#     return log_pO, p

