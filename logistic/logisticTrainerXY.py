# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

trainingSetFile = 'simTrainSetXY.pkl'

##############################################################################
##############################################################################
# load in training set
with open(trainingSetFile, 'rb') as file:
    (frames,xclassification,yclassification) = pickle.load(file)
    
# read parameters from training set dimenstions
numberSamples, yDim, xDim = frames.shape
print('{} frames, {} x {}'.format(numberSamples,yDim,xDim))

# pre-processing of frames
scaled_frames = frames/frames.max()

# reshaping of frames and loading into X, Y arrays for training
X = scaled_frames.reshape( (numberSamples, yDim*xDim) )
Yy = yclassification
Yx = xclassification

# define regression model and train!
xreg = linear_model.LogisticRegression()
xreg.fit(X,Yx)
yreg = linear_model.LogisticRegression()
yreg.fit(X,Yy)

# output results
print(xreg.coef_)
print(xreg.intercept_)
print(yreg.coef_)
print(yreg.intercept_)


