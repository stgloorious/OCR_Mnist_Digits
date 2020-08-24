import numpy
import time
from tqdm import tqdm

## Trains the neural network
#   @param neural_network The neural network object that should be trained
#   @param training_data_path Path to the .csv file that contains the training data. First entry must be label.
#
def train (neural_network, training_data_path):

    ## ACQUIRE TRAINING DATA FROM DATABASE
    print("Getting training data...")
    training_data_file = open(training_data_path,'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #debug messages
    print ("Transferred %d lines of training data into memory" %(len(training_data_list)))
    print ("")

    ## TRAIN MODEL
    print ("Training in progess...")

    # prepare progress bar
    progress=0
    maxValue=len(training_data_list)
    pbar = tqdm(total=maxValue,desc="Training")

    # go through all records in the training data
    for record in training_data_list:
        #separate by commas
        all_values = record.split(',')
        #scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        #create the target output values 
        targets = numpy.zeros(10) + 0.01
        #all_values[0] is the target label
        targets[int(all_values[0])] = 0.99
        neural_network.train(inputs,targets)
        pbar.update()
    pass  
    pbar.close()
pass