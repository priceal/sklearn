# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal

create and saves a set of frames of randomly placed Gaussian peaks. with
randomly placed centers.

a pickled database of frames with classification is saved, a 2-tuple of the
numpy array of frames and a Boolean array of T/F if center of peak is in
rectangular "hit region."

frame: left edge is -0.5 and right edge is frameDim[1]-0.5
"""

runfile('initialize.py', current_namespace=True)

# define parameters
frameDims = (7,7)   # dimensions of frame
xSpan = (-2.0,8.0)   # x range of peak positions
ySpan = (-2.0,8.0)  # y range
amp = 150.0          # amp of gaussian
sigma = 2.0         # std of gaussian
normalization = False      # normalize frame ?
numberSamples = 100000

# hit region --- where value is TRUE
xHit = [2.5,3.5]
yHit = [2.5,3.5]

# if you want to save the set
saveFile = False
if saveFile:
    saveFileName = 'testTraingSet.pkl'

##############################################################################
##############################################################################
# setup training input frames
coordinates = g.multixy(numberSamples,xr=xSpan,yr=ySpan)
frames = amp*g.multigf(coordinates,sigma,frameDims,norm=normalization)
frames = np.array(frames, dtype='int')

# now create classification for training
xClass = (coordinates[:,1]>xHit[0]) & (coordinates[:,1]<xHit[1])
yClass = (coordinates[:,0]>yHit[0]) & (coordinates[:,0]<yHit[1])
xyClass = xClass & yClass

# plot targets to check
clrs = [ 'red' if cl else 'blue' for cl in xyClass ]
plt.figure(1)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
#ax.scatter(coordinates[:,1],coordinates[:,0],c=clrs)

if saveFile:
    with open(saveFileName,'wb') as file:
        pickle.dump((frames,xyClass),file)