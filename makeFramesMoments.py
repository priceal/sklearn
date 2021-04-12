# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

# define parameters
frameDims = (7,7)   # coords go from -0.5 to dim-0.5
xSpan = (2.0,4.0)
ySpan = (2.0,4.0)
sigma = 1.5
integratedIntensity = 1.0
numberSamples = 1000

# setup dependent/independent variable lists, and errors, calculate features
x = rd.uniform(xSpan[0], xSpan[1], numberSamples)
y = rd.uniform(ySpan[0], ySpan[1], numberSamples)
frames = g.multigf(zip(x,y),sigma,integratedIntensity,frameDims)
framemoments = pa.statframes_(frames,back=False)[:,3:5]

# define regression model and arrays
reg = linear_model.LinearRegression()
X = framemoments
Y = np.array(list(zip(x,y)))
reg.fit(X,Y)

print(reg.coef_)
print(reg.intercept_)

# generate test data and features, makes predictions
x = rd.uniform(xSpan[0], xSpan[1], numberSamples)
y = rd.uniform(ySpan[0], ySpan[1], numberSamples)
testframes = g.multigf(zip(x,y),sigma,integratedIntensity,frameDims)
testX = pa.statframes_(testframes,back=False)[:,3:5]
pred = reg.predict(testX)

# plot results
plt.figure(2)
ax = plt.subplot()
for sx, sy, p in zip(x,y,pred):
    ax.set_aspect('equal')
    ax.scatter([sx,p[0]],[sy,p[1]],marker='.', c=['blue','red'])
    ax.plot([sx,p[0]],[sy,p[1]],'k-')
             
xcoefs = np.append(reg.coef_[0],reg.coef_[0].mean())
ycoefs = np.append(reg.coef_[1],reg.coef_[1].mean())


