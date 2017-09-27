# multiplyNNDemo
Demonstration of the value of a Multiply() layer when generalizing non-linearities

# Explanation
Most neural networks compose many layers of activations to model potentially strongly non-linear effects. NNs can approximate any function within a certain error bound with a finite number of parameters, but often the function fails to generalize outside the scope of the training data.

For instance consider learning `f(x) = x^2`. A NN can model this within the range `[-1, 1]`, `[-100, 100]`, etc. but when shown input outside those ranges, there is no way to have learned a purely additive function that will generalize.

But if we equip the network with layers capable of elementwise multiplication (and choose activations carefully), it has a means of capturing the relationship.

# Code
This code generates a toy example of a function `f(x1, x2) = x1^2 + x2^2`

The training input is on the range `(-10, 10)` and the test input is `(-20, 20)`. 

Network 1 has 2 Dense layers and achieves a RMSE of ~100,000 on the test set. As seen below, it does well in the training range but fails uniformly outside it (blue circles mean low RMSE, white mean some error, red mean drastic error)

![norm](https://user-images.githubusercontent.com/7809188/30895383-e7686614-a317-11e7-9afc-60c72d0ba3d7.jpg)

Network 2 has the same number of nodes but has its Dense layers connected with a keras.layers.Multiply(). It achieves an RMSE of ~1000. It does just as well within the training range, but also significantly better on the unseen range. 
![mult](https://user-images.githubusercontent.com/7809188/30895381-e38b66b8-a317-11e7-9a2a-ae9c500730dc.jpg)

