
import pyGM as gm
import numpy as np

# Make a basic model
X = [gm.Var(i,3) for i in range(7)]
factors = [ gm.Factor([X[0],X[1]],1.0) , gm.Factor([X[0],X[2]],1.0) , gm.Factor([X[1],X[2]],1.0),
            gm.Factor([X[1],X[3]],1.0) , gm.Factor([X[2],X[3]],1.0) , gm.Factor([X[3],X[4]],1.0),
            gm.Factor([X[2],X[4]],1.0) , gm.Factor([X[4],X[5]],1.0) , gm.Factor([X[2],X[5]],1.0)]

for i in range(len(factors)):               # fill the tables with random values
    factors[i].table = np.random.rand(3,3)

# Perform variable elimination
model_ve = gm.GraphModel(factors) # make a new model (will be modified by VE)

pri = [1.0 for Xi in X]
pri[1], pri[3] = 2.0,2.0          # eliminate X1 and X3 last
order = gm.eliminationOrder(model_ve, orderMethod='minfill', priority=pri)[0]
print order,'\n'

sumElim = lambda F,Xlist: F.sum(Xlist)   # helper function for eliminate
model_ve.eliminate(order[:-2], sumElim)  # eliminate all but last two

p13 = model_ve.joint()
lnZ = np.log(p13.sum())   # can get the (log) partition function as well
print 'lnZ: ',lnZ,'\n'
p13 /= p13.sum()
print p13, '\n',p13.table