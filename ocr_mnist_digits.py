#%%
import matplotlib.pyplot
import numpy
import scipy.special
import random
import progressbar
from time import sleep
import time
from tqdm import tqdm

import nn

n = nn.neuralNetwork((28*28),100,10,0.3)
print("Getting training data...")
#load mnist data into list
training_data_file = open("C:/Users/stefa/Desktop/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

print ("Transferred %d lines of training data into memory" %(len(training_data_list)))
print ("")
print ("Training in progess...")

sleep(0.5)

progress=0
maxValue=len(training_data_list)
   
# go through all records in the training data
pbar = tqdm(total=maxValue,desc="Training")
for record in training_data_list:
    #separate by commas
    all_values = record.split(',')
    #scale and shift inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    #create the target output values 
    targets = numpy.zeros(10) + 0.01
    #all_values[0] is the target label
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pbar.update()
pass  
pbar.close()

""" print ("Testing with training data...")
sleep(0.5)
error_count = 0
pTestbar = tqdm(total=maxValue,desc="Testing...")
for record in training_data_list:
    #separate by commas
    all_values = record.split(',')
    #scale and shift inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

    out = n.query(inputs)
    pTestbar.update()
    recognized_character = int(numpy.where(out == numpy.amax(out))[0])
    training_character = int(all_values[0])
    if training_character is not recognized_character:
        error_count = error_count+1
pass
pTestbar.close()
print ("Training with given dataset completed.") """

#load mnist data into list
testing_data_file = open("C:/Users/stefa/Desktop/mnist_test.csv",'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

print ("Transferred %d lines of testing data into memory" %(len(testing_data_list)))
error_count=0
pTestbar = tqdm(total=len(testing_data_list),desc="Testing")
count=0
for record in testing_data_list:
    #separate by commas
    all_values = record.split(',')
    #scale and shift inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

    out = n.query(inputs)
    pTestbar.update()
    recognized_character = int(numpy.where(out == numpy.amax(out))[0])
    unknown_character = int(all_values[0])
    if recognized_character != unknown_character:
        error_count=error_count+1
        if error_count <= 3:
            image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
            fig, (inp, outp) = matplotlib.pyplot.subplots(1,2)
            fig = matplotlib.pyplot.figure()
            inp.imshow(image_array,cmap='Greys',interpolation='None')
            inp.set_title("Input")
            outp.imshow(out,cmap='Greys',interpolation='None')
            outp.set_yticks([0,1,2,3,4,5,6,7,8,9]) 
            outp.set_title("Output")

    if count <= 10:
        image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
        fig, (inp, outp) = matplotlib.pyplot.subplots(1,2)
        fig = matplotlib.pyplot.figure()
        inp.imshow(image_array,cmap='Greys',interpolation='None')
        inp.set_title("Input")
        outp.imshow(out,cmap='Greys',interpolation='None')
        outp.set_yticks([0,1,2,3,4,5,6,7,8,9]) 
        outp.set_title("Output")
        count=count+1
pass
pTestbar.close()

print("Prediction accuracy: %.2f %% (%d of %d samples correct)" %(((len(testing_data_list)-error_count)/len(testing_data_list))*100, len(testing_data_list)-error_count, len(testing_data_list)))

# %%
