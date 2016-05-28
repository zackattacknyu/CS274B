from os import walk
datapath = 'data/';
_, _, files = next(walk(datapath), (None, None, []))

s = 50 # to load file number 50:
fh = open(datapath+files[s],'r')
rawlines = fh.readlines()
lines = [line.strip('\n').split(',') for line in rawlines]
fh.close()
ys = [int(l[1])-1 for l in lines]
xs = [[int(l[2])-1,int(l[3]),int(l[4]),int(l[5])-1,int(l[6])-1] for l in lines]