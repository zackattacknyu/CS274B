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

print

#Part B Code
Tmatrix = np.zeros((len(xvals),len(xvals)))
for i in range(mSeq):
    curSeq = x[i]
    for j in range(1,len(curSeq)):
        xPrev = curSeq[j-1]
        xCurrent = curSeq[j]
        Tmatrix[xPrev,xCurrent] += 1
Tsum = np.matrix(np.sum(Tmatrix,axis=1))
Tsum = np.transpose(Tsum)
TsumTiled = np.matlib.repmat(Tsum,1,8)
print 'Unnormalized T'
print Tmatrix[0:5,0:5]
Tmatrix = np.divide(Tmatrix,TsumTiled)
print 'Transition Matrix (first 5 states) is as follows:'
print Tmatrix[0:5,0:5]

epsilon = 1e-8
Tmatrix2 = np.matrix(Tmatrix)
curX = np.matrix(p0vals)
for i in range(500):
    prevX = np.copy(curX)
    curX = curX*Tmatrix2
    diffX = np.abs(np.subtract(prevX,curX))
    if np.sum(diffX)<epsilon:
        break
print
print 'Stationary Distribution:'
print np.transpose(np.matrix(curX))

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

#Part C Code
Omatrix = np.zeros((len(xvals),len(ovals)))
Omatrix2 = np.zeros((len(ovals),len(xvals)))
for i in range(mSeq):
    curSeq = x[i]
    curObs = o[i]
    for j in range(len(curSeq)):
        xt = curSeq[j]
        ot = curObs[j]
        Omatrix[xt,ot] += 1
        Omatrix2[ot,xt] += 1
Osum = np.matrix(np.sum(Omatrix,axis=1))
Osum = np.transpose(Osum);
OsumTiled = np.matlib.repmat(Osum,1,20)
Omatrix = np.divide(Omatrix,OsumTiled)
print
print 'Emission Probability Matrix (first 5 states) is as follows:'
print Omatrix[0:5,0:5]

Osum2 = np.matrix(np.sum(Omatrix2,axis=1))
Osum2 = np.transpose(Osum2);
OsumTiled2 = np.matlib.repmat(Osum2,1,8)
Omatrix2 = np.divide(Omatrix2,OsumTiled2)
#print np.matrix(np.sum(Omatrix2,axis=1))
#print 'Ob matrix (first 5 states) is as follows:'
#print Omatrix2[0:5,0:5]

p0 = np.zeros(len(ovals))
for i in range(mSeq):
    curSeq = o[i]
    x0val = curSeq[0]
    p0[x0val] += 1
p0 = np.matrix(np.divide(p0, mSeq))
p0 = np.transpose(p0)

Ob = np.matrix(np.copy(Omatrix))
dx,do2 = Ob.shape   # if a numpy matrix
curT = 0
curObs = o[curT]
L = len(curObs)
f = np.zeros((L,dx))
r = np.zeros((L,dx))
p = np.zeros((L,dx))


p0col = np.reshape(p0vals,(8,1))
compF = np.multiply(Ob[:,curObs[0]],p0col)
f[0,:] = np.reshape(compF,8)   # compute initial forward message
log_pO = np.log(f[0,:].sum())  # update probability of sequence so far
f[0,:] /= f[0,:].sum()  # normalize (to match definition of f)

#
# curF = np.reshape(f[0,:],(1,8))
# curXprobs = np.transpose(curF*Tmatrix)
# curObcol = Ob[:,curObs[1]]
# print curXprobs.shape
# print curObcol.shape
# f[1,:] = np.reshape(np.multiply(curXprobs,curObcol),8)
# f[1,:] /= f[1,:].sum()  # normalize (to match definition of f)
# print f[1,:]

for t in range(1,L):    # compute forward messages
    prevF = np.reshape(f[t-1,:], (1,dx))
    curXprobs = np.transpose(prevF * Tmatrix)
    curObcol = Ob[:, curObs[t]]
    f[t,:] = np.reshape(np.multiply(curXprobs, curObcol), dx)
    log_pO += np.log(f[t,:].sum())
    f[t,:] /= f[t,:].sum()  # normalize (to match definition of f)

print
print 'First 5 F rows'
print f[0:5,:]
print
print 'Log Likelihood:'
print log_pO

r[L-1,:] = 1.0  # initialize reverse messages
p[L-1,:] = np.multiply(r[L-1,:],f[L-1,:])  # and marginals
print p[L-1,:]
p[L-1,:] /= p[L-1,:].sum()

for t in range(L-2,-1,-1):
    prevR = np.reshape(r[t+1,:],(dx,1))
    curObcol = Ob[:, curObs[t+1]]
    curCol = np.matrix(np.multiply(prevR,curObcol))
    r[t,:] = np.reshape(Tmatrix*curCol, dx)
    r[t,:] /= r[t,:].sum()
    p[t,:] = np.multiply(r[t,:],f[t,:])
    p[t,:] /= p[t,:].sum()

print 'time 6 reverse messages:',r[6,:]
print 'time 6 forward messages:', f[6,:]

print 'time 6 marginal probs'
print p[6,:]

print 'First 5 Files:'
for i in range(0,5):
    print filenames[i]

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

