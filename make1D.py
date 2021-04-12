# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

# define parameters
frameDim = 2  # coords go from -0.5 to dim-0.5
xSpan = (0.0,1.0)

sigma = 1.0
integratedIntensity = 1.0
numberSamples = 1000


# setup dependent/independent variable lists, and errors
x = rd.uniform(xSpan[0], xSpan[1], numberSamples)
y = np.ones(numberSamples)*(frameDim-1.0)/2.0
frames = g.multigf(zip(x,y),sigma,integratedIntensity,(frameDim,frameDim))

lines = frames.sum(axis=2)

ldf = pd.DataFrame(data=lines,columns=['1','2'])

sbs.pairplot(ldf)