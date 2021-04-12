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
numberSamples = 1000


# setup dependent/independent variable lists, and errors
x = np.random.uniform(xSpan[0], xSpan[1], numberSamples)
y = np.random.uniform(ySpan[0], ySpan[1], numberSamples)
frames = g.multigf(zip(x,y),sigma,frameDims,norm=normalization)

# define regression model and arrays
reg = linear_model.LinearRegression()
X = frames.reshape((numberSamples,frameDims[0]*frameDims[1]))
Y = np.array(list(zip(x,y)))
reg.fit(X,Y)

print(reg.coef_)
print(reg.intercept_)
#plt.plot(x,y,'o')

#plt.plot(pred[:,0],pred[:,1],'*')
#xSpan = (0.0,6.0)
#ySpan = (0.0,6.0)

x = np.random.uniform(xSpan[0], xSpan[1], numberSamples)
y = np.random.uniform(ySpan[0], ySpan[1], numberSamples)
testframes = g.multigf(zip(x,y),sigma,frameDims,norm=normalization)
testX = testframes.reshape((numberSamples,frameDims[0]*frameDims[1]))
pred = reg.predict(testX)

plt.figure(1)
for sx, sy, p in zip(x,y,pred):
    plt.plot([sx,p[0]],[sy,p[1]],'bo-')
             
#xcoefs = np.append(reg.coef_[0].mean(),reg.coef_[0])
#ycoefs = np.append(reg.coef_[1].mean(),reg.coef_[1])

xcoefs = reg.coef_[0]
ycoefs = reg.coef_[1]

plt.figure(2)
plt.imshow(xcoefs.reshape(frameDims))
plt.figure(3)
plt.imshow(ycoefs.reshape(frameDims))
