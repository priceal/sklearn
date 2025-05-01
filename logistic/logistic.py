# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

# define parameters
frameDims = (7,7)   # coords go from -0.5 to dim-0.5
xSpan = (1.0,5.0)
ySpan = (1.0,5.0)
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

plt.figure(1)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(x,y,c=clrs)

# define regression model and arrays
reg = linear_model.LogisticRegression()
X = frames.reshape((numberSamples,frameDims[0]*frameDims[1]))
Y = xyclass
reg.fit(X,Y)


print(reg.coef_)
print(reg.intercept_)

# generate test data and features, makes predictions
x = rd.uniform(xSpan[0], xSpan[1], numberSamples)
y = rd.uniform(ySpan[0], ySpan[1], numberSamples)
testframes = g.multigf(zip(x,y),sigma,frameDims,norm=normalization)
testX = testframes.reshape((numberSamples,frameDims[0]*frameDims[1]))
testpred = reg.predict(testX)

clrs = ['red' if p else 'blue' for p in testpred]
plt.figure(2)
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(x,y,c=clrs)
