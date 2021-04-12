# -*- coding: utf-8 -*-
"""


Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

# define parameters
slope = -5.3
intercept = -26
numberSamples = 50
xMin, xMax = -13.0, 13.0
xSigma, ysigma = 3.5, 2.8

# setup dependent/independent variable lists, and errors
x = np.random.uniform(xMin, xMax, numberSamples)
y = x*slope + intercept
x += np.random.normal(0, xSigma, numberSamples)
y += np.random.normal(0, ySigma, numberSamples)

# define regression model and arrays
reg = linear_model.LinearRegression()
X = x[:,np.newaxis]   # must add new axis to x since 1 dim
Y = y
reg.fit(X,Y)

xModel = (xMin,xMax)
yModel = (xMin*reg.coef_[0]+reg.intercept_,xMax*reg.coef_[0]+reg.intercept_)

modelLabel = 'm={:2.2f}, b={:2.2f}'.format(reg.coef_[0],reg.intercept_)
plt.plot(X[:,0],Y,'.',label='data')
plt.plot(xModel,yModel,'-',label=modelLabel)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
print('The intecept is {} and the slope is {}.'.format(reg.intercept_,reg.coef_[0]))






