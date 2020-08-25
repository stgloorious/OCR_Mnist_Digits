## @package test
# Takes test data provided via a local path and compares the networks output with the labels of the test data. Provides status messages and gives performance reading at the end
#
# Test data is supposed to be in .csv format, where every digit is represented by a new line.
# Pixel values are in greyscale between 0 and 255. First value of every new line (new digit) is 
# its label (value that it is supposed to be representing).
#

import matplotlib.pyplot
import numpy
import scipy.special
import time
from tqdm import tqdm

##  @brief Tests the neural network
#   @param neural_network Neural network object that should be tested
#   @param test_data_path Path to the .csv file with the testing data. First entry must be label.
#
def test (neural_network,test_data_path):

    ## LOAD TESTING DATA
    testing_data_file = open(test_data_path,'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()
    total_count=len(testing_data_list)
    print ("Transferred %d lines of testing data into memory" %(total_count))

    # restoring data of previously trained network
    neural_network.restoreState()

    # prepare progress bar
    pbar = tqdm(total=total_count,desc="Testing")

    # keep track of how many digits the model got wrong
    error_count=0

    #iterate through all samples in the data set
    for record in testing_data_list:
        #separate by commas
        all_values = record.split(',')
        #scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01 # range is 0.01 ... 0.99
        # let the neural network determine what digit it sees
        out = neural_network.query(inputs)
        pbar.update()
        recognized_character = int(numpy.where(out == numpy.amax(out))[0])
        character_label = int(all_values[0]) # first element is label
        # check if neural network was correct
        if recognized_character != character_label:
            error_count=error_count+1
            #display the first three digits it got wrong
            if error_count <= 3:
                image_array = numpy.asfarray(all_values[1:]).reshape((28,28)) # take the data and draw it
                fig, (inp, outp) = matplotlib.pyplot.subplots(1,2) #create a figure with two subplots in it
                fig = matplotlib.pyplot.figure()
                inp.imshow(image_array,cmap='Greys',interpolation='None') # use grayscale color map to draw image
                inp.set_title("Input")
                outp.imshow(out,cmap='Greys',interpolation='None') # use grayscale to draw output plot
                outp.set_yticks([0,1,2,3,4,5,6,7,8,9]) # make sure all yticks are drawn on the output plot
                outp.set_title("Output")
    pass
    pbar.close()

    ## calculate the accuracy of the model
    correct_count = total_count-error_count
    correct_percentage = (correct_count/total_count)*100
    print("Prediction accuracy: %.2f %% (%d of %d samples correct)" %(correct_percentage,correct_count,total_count))
pass