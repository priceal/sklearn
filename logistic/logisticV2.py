# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

# define parameters
frameDims = (7,7)   # coords go from -0.5 to dim-0.5
xSpan = (0.0,6.0)
ySpan = (0.0,6.0)
sigma = 1.0
normalization = False
numberSamples = 4000

# hit region --- where value is TRUE
xrange = [2.5,3.5]
yrange = [2.5,3.5]

# setup dependent/independent variable lists, and errors, calculate features
x = np.random.uniform(xSpan[0], xSpan[1], numberSamples)
y = np.random.uniform(ySpan[0], ySpan[1], numberSamples)
frames = g.multigf(zip(x,y),sigma,frameDims,norm=normalization)
xclass = (x>xrange[0]) & (x<xrange[1])
yclass = (y>yrange[0]) & (y<yrange[1])
xyclass = xclass & yclass
clrs = [ 'red' if cl else 'blue' for cl in xyclass ]
xclrs = [ 'green' if cl else 'blue' for cl in xclass ]
yclrs = [ 'yellow' if cl else 'blue' for cl in yclass ]

plt.figure(1)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(x,y,c=clrs)

X = frames.reshape((numberSamples,frameDims[0]*frameDims[1]))

# define regression model and arrays
xreg = linear_model.LogisticRegression()
Y = xclass
xreg.fit(X,Y)

# define regression model and arrays
yreg = linear_model.LogisticRegression()
Y = yclass
yreg.fit(X,Y)

# generate test data and features, makes predictions
numberSamples = 10000
x = np.random.uniform(xSpan[0], xSpan[1], numberSamples)
y = np.random.uniform(ySpan[0], ySpan[1], numberSamples)
testframes = g.multigf(zip(x,y),sigma,frameDims,norm=normalization)
testX = testframes.reshape((numberSamples,frameDims[0]*frameDims[1]))
xpred = xreg.predict(testX)
ypred = yreg.predict(testX)
testpred = xpred & ypred

clrs = ['red' if p else 'blue' for p in testpred]
plt.figure(2)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(x,y,c=clrs)
