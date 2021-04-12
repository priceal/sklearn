# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

# define parameters
frameDim = (5,5)  # coords go from -0.5 to dim-0.5
xSpan = (0.0,4.0)
ySpan = (0.0,4.0)

sigma = 1.5
integratedIntensity = 1.0
numberSamples = 1000


# setup dependent/independent variable lists, and errors
x = rd.uniform(xSpan[0], xSpan[1], numberSamples)
y = rd.uniform(ySpan[0], ySpan[1], numberSamples)
frames = g.multigf(zip(x,y),sigma,integratedIntensity,frameDim)

lines = frames.reshape(numberSamples,frameDim[0]*frameDim[1])

ldf = pd.DataFrame(data=lines,columns=['11','12','13','21','22','23','31','32','33'])

sbs.pairplot(ldf)