import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape x_train and x_test to flatten the 28x28 images into a 784-element vector
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

assert x_train.shape == (60000, 784)
assert x_test.shape == (10000, 784)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def sig(x):
    #Defining the sigmoid function
    return 1/(1+np.exp(-x))
def sigd(x):
    s = sig(x)
    return s * (1 - s)
def random(x, y):
    return 2*(np.random.rand(x, y))-1

def softm(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=0, keepdims=True)
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6/(fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))
input_size = 784
data_size = 60000

hl_total = int(input_size*(2/3))       #Total number of nodes in hidden layer 1
hl2_total = int(hl_total*(2/3))        #Total number of nodes in hidden layer 2
output_number = 10
weights_array1 = xavier_init(input_size, hl_total)
weights_array2 = xavier_init(hl_total, hl2_total)
weights_array3 = xavier_init(hl2_total, output_number)

bias_array1 = np.random.rand(hl_total, 1)                          #Biases for the first hidden layer connections

bias_array2 = np.random.rand(hl2_total, 1)                         #Biases for hidden layer 2

bias_array3 = np.random.rand(output_number, 1)                                     #Biases for output
layer = 3
#Training#
gradient = []
batch_grad = []
prevbatch = 0
def train(epoch_number, batch_size, dataset_len):
    for e in range (epoch_number):
    
        perm = np.random.permutation(dataset_len)
        x_train[:] = x_train[perm]
        y_train[:] = y_train[perm]

        for j in range (dataset_len//batch_size):
            wl1 = np.zeros_like(weights_array1)
            wl2 = np.zeros_like(weights_array2)
            wl3 = np.zeros_like(weights_array3)

            bl1 = np.zeros_like(bias_array1)
            bl2 = np.zeros_like(bias_array2)
            bl3 = np.zeros_like(bias_array3)
            for i in range (batch_size):
                index = j*batch_size+i
                #Initialize#
                target_vector = np.zeros((output_number, 1))
                #FORWARD-PROPOGATION#

                array_input = x_train[index].reshape(-1,1)              #Input array that the nueral network takes in

                zL1 = (np.dot(weights_array1, array_input))+(bias_array1)
                hl_val = sig(zL1)  #Values for hidden layer 1

                zL2 = (np.dot(weights_array2, hl_val)+(bias_array2))
                hl2_val = sig(zL2)      #Values for hidden layer 2

                zL3 = (np.dot(weights_array3, hl2_val))+(bias_array3)
                output_val = softm(zL3)  #output values
                #BACK-PROPOGATION#
                target_vector[y_train[index]]=1
                delta3 = output_val-target_vector 
                wl3 += np.dot(delta3,hl2_val.T)
                bl3 += delta3
                delta2 = np.dot(delta3, weights_array3.T)*sigd(zL2)
                wl2 += np.dot(delta2,hl_val.T)
                bl2 += delta2
                delta1 = np.dot(delta2, weights_array2.T)* sigd(zL1)
                wl1 += np.dot(delta1,array_input.T)
                bl1 += delta1

            wl3 = wl3/batch_size
            bl3 = bl3/batch_size
            wl2 = wl2/batch_size
            bl2 = bl2/batch_size
            wl1 = wl1/batch_size
            bl1 = bl1/batch_size
            learning_rate = 0.01

            weights_array3 -= learning_rate * wl3
            bias_array3 -= learning_rate * bl3

            weights_array2 -= learning_rate * wl2
            bias_array2 -= learning_rate * bl2

            weights_array1 -= learning_rate * wl1
            bias_array1 -= learning_rate * bl1
train(50, 32, 60000)
print("Done")
index2 = input("Select random index: ")
print(y_test[index2])

array_input = x_train[index2].reshape(-1,1)              #Input array that the nueral network takes in

zL1 = (np.dot(weights_array1, array_input))+(bias_array1)
hl_val = sig(zL1)  #Values for hidden layer 1

zL2 = (np.dot(weights_array2, hl_val)+(bias_array2))
hl2_val = sig(zL2)      #Values for hidden layer 2

zL3 = (np.dot(weights_array3, hl2_val))+(bias_array3)
output_val = softm(zL3)  #output values
print(max(output_val))
