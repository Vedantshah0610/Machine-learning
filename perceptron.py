import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100,learning_rate = 0.001):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        
    def predict(self,inputs):
        summation = np.dot(inputs,self.weights[1:]) + self.weights[0] #w.x + b
        if summation>0:
            activation=1
        else:
            activation=0
        return activation
    
    def train(self,training_inputs,labels):
        e_min=100.00
        epoch=0
        for i in range(self.threshold):
            e_max=0.00
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error=self.learning_rate * (label-prediction)
                self.weights[1:]+= error * inputs
                self.weights[0] += error
                e_max=max(e_max,error)
            if(e_min>e_max):
                e_min=e_max
                epoch=i
        print("epoch=",epoch)      


