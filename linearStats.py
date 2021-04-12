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
    stats
except NameError:
    print("importing modules")
    import random
    import pylab as plt
    from sklearn import linear_model
    import numpy as np
    import statistics as stats

# define parameters
slope = 1.25
intercept = 3.5
xMin = 0.0
xMax = 10.0
numberSamples = 12
xSigma = 0.4
ySigma = 0.0
numberSamples = 1000

slopeList = []
interceptList = []
for sample in range(numberSamples):
    
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

    slopeList.append(reg.coef_[0])
    interceptList.append(reg.intercept_)
    
    
fig, ax = plt.subplots(1,2)
ax[0].hist(slopeList)
ax[0].set_xlabel('slope')
ax[0].set_ylabel('frequency')
ax[1].hist(interceptList)
ax[1].set_xlabel('intercept')
ax[1].set_ylabel('frequency')
    
print('mean slope +/= STD = {:2.2f} +/- {:2.2f}'.format(stats.mean(slopeList),stats.stdev(slopeList)) )
print('mean intercept +/= STD = {:2.2f} +/- {:2.2f}'.format(stats.mean(interceptList),stats.stdev(interceptList)) )

   





