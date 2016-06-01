from os import walk
import pyGM as gm
import numpy as np
import pyGM.wmb
#import matplotlib.pyplot as plt
#import networkx as nx

datapath = 'data/'
_, _, files = next(walk(datapath), (None, None, []))

s = 50 # to load file number 50:
fh = open(datapath+files[s],'r')
rawlines = fh.readlines()
lines = [line.strip('\n').split(',') for line in rawlines]
fh.close()
ys = [int(l[1])-1 for l in lines]
xs = [[int(l[2])-1,int(l[3]),int(l[4]),int(l[5])-1,int(l[6])-1] for l in lines]


feature_sizes = [1,2,2,201,201]
ThetaF = [.001*np.random.rand(10,feature_sizes[f]) for f in range(len(feature_sizes))]
ThetaP = .001*np.random.rand(10,10)
Loss = 1.0 - np.eye(10) # hamming loss

print len(ThetaF)
print len(ThetaP)

print Loss[:,1]

num_iter = 1


# step size, etc.
for iter in range(num_iter):
    #for s in np.random.permutation(len(files)):
    for s in range(5):
        # Load data ys,xs
        fh = open(datapath + files[s], 'r')
        rawlines = fh.readlines()
        lines = [line.strip('\n').split(',') for line in rawlines]
        fh.close()
        ys = [int(l[1]) - 1 for l in lines]
        xs = [[int(l[2]) - 1, int(l[3]), int(l[4]), int(l[5]) - 1, int(l[6]) - 1] for l in lines]

        ns = len(ys)

        #Define random variables for the inference process:
        Y = [gm.Var(i,10) for i in range(ns)]

        factors = []
        for ii in range(ns):
            curTable = np.matrix(np.zeros((10,1)))

            for ff in range(len(feature_sizes)):
                curTh = np.matrix(ThetaF[ff])
                curX = xs[ii][ff]
                curMat = np.matrix(curTh[:,curX])
                curTable = np.add(curTable,curMat)

            if ii<(ns-1):
                curMat = np.matrix(ThetaP[:,ys[ii+1]])
                curMat = np.reshape(curMat,(10,1))
                curTable = np.add(curTable,curMat)

            factors.append(gm.Factor([Y[ii]]))
            factors[ii].table = np.matrix(np.exp(curTable))

        model_pred = gm.GraphModel(factors)

        # Copy factors and add extra Hamming factors for loss-augmented model
        factors_aug = [ f for f in factors ]
        factors_aug.extend( [gm.Factor([Y[i]], Loss[:,ys[i]]).exp() for i in range(ns)] )
        model_aug = gm.GraphModel(factors_aug)

        order = range(ns) # eliminate in sequence (Markov chain on y)
        wt = 1e-4 # for max elimination in JTree implementation

        # Now, the most likely configuration of the prediction model (for prediction) is:
        yhat_pred = gm.wmb.JTree(model_pred,order,wt).argmax()
        #print yhat_pred

        # and the maximizing argument of the loss (for computing the gradient) is
        yhat_aug = gm.wmb.JTree(model_aug,order,wt).argmax()
        #print yhat_aug



        # use yhat_pred & ys to keep a running estimate of your prediction accuracy & print it
        #... # how often etc is up to you
        print ns
        acc = 0
        yhatVals = yhat_pred.values()
        for kk in range(ns):
            if(ys[kk]-yhatVals[kk] < 1):
                acc += 1
        acc = np.divide(np.double(acc),np.double(ns))
        print acc
        print '----------'


        # use yhat_aug & ys to update your parameters theta in the negative gradient direction
        #...
