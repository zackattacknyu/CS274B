from os import walk
import pyGM as gm
import numpy as np
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

        curTh = np.matrix(ThetaF[0])
        curX = np.matrix(xs[s][0])
        newMat = np.multiply(curTh,curX)
        print newMat.shape
        #factors = [gm.Factor(Y[0],ThetaF[ff].dot(xs[s][ff])) for ff in range(5)]


#ns = len(ys)

# Build "prediction model" using your parameters
#factors = [ ...
# don't forget pyGM expects models to be products of factors,
# so exponentiate the factors before making a model...
#model_pred = gm.GraphModel(factors);
# Copy factors and add extra Hamming factors for loss-augmented model
#factors_aug = [ f for f in factors ]
#factors_aug.extend( [gm.Factor([Y[i]], Loss[:,ys[i]]).exp() for i in range(n)] );
#model_aug = gm.GraphModel(factors_aug);
#order = range(n); # eliminate in sequence (Markov chain on y)
#wt = 1e-4; # for max elimination in JTree implementation
# Now, the most likely configuration of the prediction model (for prediction) is:
#yhat_pred = gm.wmb.JTree(model_pred,order,wt).argmax();
# and the maximizing argument of the loss (for computing the gradient) is
#yhat_aug = gm.wmb.JTree(model_aug,order,wt).argmax();
# use yhat_pred & ys to keep a running estimate of your prediction accuracy & print it
#... # how often etc is up to you
# use yhat_aug & ys to update your parameters theta in the negative gradient direction
#...
