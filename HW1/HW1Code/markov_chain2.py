import numpy as np
from os import walk


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
print p0vals


#Part B Code
Tmatrix = np.zeros((len(xvals),len(xvals)))
numEx = 0
for i in range(mSeq):
    curSeq = x[i]
    for j in range(1,len(curSeq)):
        xPrev = curSeq[j-1]
        xCurrent = curSeq[j]
        Tmatrix[xPrev,xCurrent] += 1
        numEx+=1
Tmatrix = np.divide(Tmatrix,numEx)
print
print 'Transition Matrix (first 5 states) is as follows:'
print Tmatrix[0:4,0:4]


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

