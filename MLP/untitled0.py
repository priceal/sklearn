#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:15:40 2025

@author: allen
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

clf=MLPClassifier(solver='lbfgs',hidden_layer_sizes=(3,2), \
                  random_state=1)

clf.fit(X,y)

p=clf.predict(X)
p
p-y
