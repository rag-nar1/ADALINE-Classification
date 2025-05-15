# updating weights after each iteration in adaline
# updating weights after each input in perceptron

# Error = (Y - Y_hat)^2  where Y is the target and Y_hat is the predicted value
# dE/ dY = 2 * (Y - Y_hat) , dY/dW = X -> Y = WX + B, dE/dW = dE/dY * dY/dW
# d(Error)/d(W) = 2 * (Y - Y_hat) * X where W is the weights and X is the input
# W = W + learning_rate * d(Error)/d(W) = W + learning_rate * 2 * (Y - Y_hat) * X
# MSE = (Y - Y_hat)^2 / N where N is the number of inputs and MSE is the mean squared error
# bias = bias + error * learning_rate
# repeat for some number of iterations
# ! in adaline all variables are vectors not a single value


import random
import numpy as np

class Adaline:

    def __init__(self, num_inputs, learning_rate=0.01, iterations=100):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.bias = 1
        self.errors = []
        self.init_weights()
        self.classification_rate = []
     
    def threshold(self, input) -> int: # unit step function
        if input >= 0:
            return 1
        return -1
    
    def init_weights(self): # random initial weights
        self.weights = np.array([random.random() for _ in range(self.num_inputs)])

    def multiply(self, inputs) -> float: # dot product of inputs and weights
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        return output
    
    def calculate_classification_rate(self, Y_hat, Y) -> float:
        return np.mean(Y_hat == Y)
    
    def update_weights(self, X , Y_hat , Y) -> None:
        errors = 2 * (Y - Y_hat)
        self.weights[:] += self.learning_rate * X.T.dot(errors)
        self.bias += self.learning_rate * errors.sum()
            
    
    def train(self, X , Y):
        for _ in range(self.iterations):
            Y_hat = np.array([])
            for i in range(len(X)):
                Y_hat = np.append(Y_hat , self.threshold(self.multiply(X[i]) + self.bias))
            
            self.update_weights(X , Y_hat , Y)  
            self.classification_rate.append(self.calculate_classification_rate(Y_hat , Y) * 100)
    
    def predict(self , X):
        return self.threshold(self.multiply(X) + self.bias)
        


