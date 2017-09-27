# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 22:38:01 2017

@author: jmf
"""

import numpy as np
import sklearn.metrics as met
import keras
from keras.layers import Dense, Merge, Input
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import cm

# define range of valid input and number of examples
trainExamples = 5000
testExamples = 25
trainMin = -10
trainMax = 10
testMin = -20
testMax = 20
noise = 0.1

# generate training data for f(x,y) = x**2 + y**2
Xtrain = np.random.rand(trainExamples, 2) 
ytrain = Xtrain[:,0]**2 + Xtrain[:,1]**2
Xtrain += noise*np.random.rand(trainExamples, 2)

# generate test data for the wider range
testx = testy = np.linspace(testMin, testMax, testExamples)
Xtest = []
xs = []
ys = []
for x in testx:
    for y in testy:
        Xtest.append([x,y])
        xs.append(x)
        ys.append(y)

Xtest = np.array(Xtest)
ytest = Xtest[:,0]**2 + Xtest[:,1]**2

inp1 = Input(shape=(2,))
d1 = Dense(32)(inp1)
d1 = Dense(32)(d1)
d1 = Dense(1)(d1)
model1 = Model(inp1, d1)
model1.compile(optimizer='rmsprop', loss='mean_squared_error')

model1.fit(Xtrain, ytrain, epochs=2)

preds = model1.predict(Xtest)
rmse1 = met.mean_squared_error(ytest,preds)
print(rmse1)

# repeat the procedure allowing for a dot product
inp2 = Input(shape=(2,))
d2 = Dense(16)(inp2)
d2b = Dense(16)(inp2)
m = keras.layers.Multiply()([d2, d2b])
m = Dense(32)(m)
f = Dense(1)(m)
model2 = Model(inp2, f)
model2.compile(optimizer='rmsprop', loss='mean_squared_error')

model2.fit(Xtrain, ytrain, epochs=2)

preds2 = model2.predict(Xtest)
rmse2 = met.mean_squared_error(ytest,preds2)
print(rmse2)

normDiffs = np.abs(ytest - preds)
multDiffs = np.abs(ytest - preds2)
diffMax = np.max([np.max(normDiffs),np.max(multDiffs)])

plt.figure(0)
plt.scatter(xs, ys, c=normDiffs/diffMax, cmap = cm.coolwarm,
            s=30)
plt.savefig('norm.jpg')

#plt.figure(1)
#plt.scatter(xs, ys, multDiffs/diffMax)
#plt.savefig('mult.jpg')