import numpy as np

f12 = np.ones((3,3))
f13 = np.ones((3,3))
f23 = np.ones((3,3))

prod1 = f13.dot(np.transpose(f23))
p12Init = np.multiply(f12,prod1)
z12 = np.sum(p12Init,axis=None)
p12 = np.divide(p12Init,z12)
print p12

phat12 = [[0.249,0.002,0.311],[0.017,0.024,0.015],[0.029,0.348,0.000]]
print phat12

newF12 = np.multiply(f12,np.divide(phat12,p12))
print newF12

prod2 = newF12.dot(f23)
p13Init = np.multiply(f13,prod2)
z13 = np.sum(p13Init,axis=None)
p13 = np.divide(p13Init,z13)
print p13


phat13 = [[0.136,0.118,0.309],[0.003,0.029,0.025],[0.0348,0.014,0.015]]
newF13 = np.multiply(f13,np.divide(phat13,p13))

print newF13