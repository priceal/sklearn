# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

# define parameters
frameDims = (5,5)   # dimensions of frame
xSpan = (0.0,4.0)   # x range of peak positions
ySpan = (0.0,4.0)   # y range
amp = 200.0          # amp of gaussian
sigma = 1.0         # std of gaussian
normalization = False      # normalize frame ?
numberSamples = 10000

# hit region --- where value is TRUE
xrange = [1.5,2.5]
yrange = [1.5,2.5]

saveFileName = 'simTrainSetXY.pkl'

##############################################################################
##############################################################################
# setup training input frames
coordinates = g.multixy(numberSamples,xr=xSpan,yr=ySpan)
frames = amp*g.multigf(coordinates,sigma,frameDims,norm=normalization)
frames = np.array(frames, dtype='int')

# now create classification for training
xclass = (coordinates[:,1]>xrange[0]) & (coordinates[:,1]<xrange[1])
yclass = (coordinates[:,0]>yrange[0]) & (coordinates[:,0]<yrange[1])
xyclass = xclass & yclass

# plot targets to check
clrs = [ 'red' if cl else 'blue' for cl in xyclass ]
plt.figure(1)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(coordinates[:,1],coordinates[:,0],c=clrs)

with open(saveFileName,'wb') as file:
    pickle.dump((frames,xclass,yclass),file)