# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)


##############################################################################
##############################################################################
# read parameters from training set dimenstions
totalSamples, yDim, xDim = frames.shape
print('all data: {} frames, {} x {}'.format(totalSamples,yDim,xDim))

# pre-processing of frames
scaled_frames = frames/frames.max()
X = scaled_frames.reshape( (len(frames), yDim*xDim) )
y = xyClass

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, shuffle=True)    
    
# read parameters from training set dimenstions
print('training set: {} frames'.format(len(y_train)))

# read parameters from training set dimenstions
print('testing set: {} frames'.format(len(y_test)))

# define regression model and train!
reg = linear_model.LogisticRegression()
reg.fit(X_train,y_train)

# output results
print(reg.coef_)
print(reg.intercept_)

#calc statistics
y_pred = reg.predict(X_test)
TruePositives = (y_test & y_pred).sum()
FalsePositives = (~y_test & y_pred).sum()
TrueNegatives = (~y_test & ~y_pred).sum()
FalseNegatives = (y_test & ~y_pred).sum()


print('\n\tActual True\tActual False\ttotal')
print('positive\t{}\t\t{}\t\t{}'.\
      format(TruePositives,FalsePositives,TruePositives+FalsePositives))
print('negative\t{}\t\t{}\t\t{}'.\
      format(FalseNegatives,TrueNegatives,FalseNegatives+TrueNegatives))
print('totals\t{}\t\t{}\t\t{}'.\
      format(TruePositives+FalseNegatives,FalsePositives+TrueNegatives,\
             len(y_test)))
    
print('\n', classification_report(y_test,y_pred))    
    
    