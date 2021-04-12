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
    mean_squared_error
    r2_score
    
except NameError:
    print("importing modules")
    import random
    import pylab as plt
    from sklearn import linear_model
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score

# define parameters of simulated data (y = polynomial of x)
alist = [ 3.5, 1.25, -1.8, 0.7, 0.3] # ground truth coefficients
xMin, xMax = -10.0, 10.0 # range of x values
numberSamples = 30
xSigma, ySigma = 0.4, 0.2 # gaussian errors added to x and y

# order of features for regression analysis
regOrder = 2  # max order of features

# setup dependent/independent variable lists, and errors
x = [ random.uniform(xMin,xMax) for i in range(numberSamples)]
y = [ sum([ a*z**n for a,n in zip(alist,range(len(alist))) ]) for z in x]

# calculate and add errors
dx = [ random.gauss(0,xSigma) for i in range(numberSamples)]
dy = [ random.gauss(0,ySigma) for i in range(numberSamples)]
xe = [ z+dz for z,dz in zip(x,dx)]
ye = [ z+dz for z,dz in zip(y,dy)]

# setup descriptors or predictors
X = np.array([ [z**n for n in range(1,regOrder+1)] for z in xe ])
Y = np.array(y)

# define regression model and fit
reg = linear_model.LinearRegression()
reg.fit(X,Y)

#define some variables for plotting
modelCoef = reg.coef_
modelCoef = np.insert(modelCoef,0,reg.intercept_)
xModel = np.linspace(xMin,xMax,50)
yModel = [ sum([ a*z**n for a,n in zip(modelCoef,range(len(modelCoef))) ]) for z in xModel]
yPrediction = [ sum([ a*z**n for a,n in zip(modelCoef,range(len(modelCoef))) ]) for z in xe]
mse = mean_squared_error(ye,yPrediction)
r2s = r2_score(ye,yPrediction)


#modelLabel = str
plt.plot(xe,ye,'.',label=None)
plt.plot(xModel,yModel,'-',label=None)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
print('ground truth =', alist)
print('regression =', modelCoef)
print('mean squared error = {:2.2f}'.format(mse))
print('R2 = {:2.2f}'.format(r2s))





