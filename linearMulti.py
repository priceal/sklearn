# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

try:
    print('testing name definitions')
    random
    plt
    linear_model
    np
except NameError:
    print("importing modules")
    import random
    import pylab as plt
    from sklearn import linear_model
    import numpy as np

# define parameters
slope = 1.25
intercept = 3.5
xMin = 0.0
xMax = 10.0
numberSamples = 12
xSigma = 0.4
ySigma = 0.4

fig,ax = plt.subplots(3,3)
axFlat = ax.flatten()
for sample in range(9):
    
    # setup dependent/independent variable lists, and errors
    x = [ random.uniform(xMin,xMax) for i in range(numberSamples)]
    y = [ z*slope+intercept for z in x]
    dx = [ random.gauss(0,xSigma) for i in range(numberSamples)]
    dy = [ random.gauss(0,ySigma) for i in range(numberSamples)]

    # define regression model and arrays
    reg = linear_model.LinearRegression()
    X = np.array(x)[:,np.newaxis] + np.array(dx)[:,np.newaxis]
    Y = np.array(y) + np.array(dy)
    reg.fit(X,Y)

    modelSlope = reg.coef_[0]
    modelIntercept = reg. intercept_
    xModel = np.linspace(xMin,xMax,100)
    yModel = xModel*modelSlope + modelIntercept

    modelLabel = 'm={:2.2f}, b={:2.2f}'.format(modelSlope,modelIntercept)
    axFlat[sample].plot(X[:,0],Y,'.',label=None)
    axFlat[sample].plot(xModel,yModel,'-',label=modelLabel)
    axFlat[sample].set_xlabel('x')
    axFlat[sample].set_ylabel('y')
    axFlat[sample].legend()
    
plt.show()
    
   





