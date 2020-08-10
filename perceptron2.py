import numpy as np
from perceptron import Perceptron

training_inputs=[]
training_inputs.append(np.array([1,1,1]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,0,1]))
training_inputs.append(np.array([0,0,1]))
training_inputs.append(np.array([1,1,0]))
training_inputs.append(np.array([0,1,0]))
labels=np.array([1,0,0,0,0,0])

perceptron=Perceptron(3)
perceptron.train(training_inputs,labels)

inputs=np.array([1,0,0])
print(perceptron.predict(inputs))