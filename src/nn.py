## @package nn Contains the fundamental methods to train and use a neural network. 
# 
# It is hard-coded for one hidden layer. Error back-propagation based on weights is used to train the model.
# The number of input, hidden and output layers as well as the learning rate is variable 
# and set through arguments. 
#

import numpy
import scipy.special

import train
import test

## @package nn Contains the fundamental methods to train and use a neural network. 
# 
# It is hard-coded for one hidden layer. Error back-propagation based on weights is used to train the model.
# The number of input, hidden and output layers as well as the learning rate is variable 
# and set through arguments. 
#
class neuralNetwork:
    ## Activation is set to the sigmoid function, rest can be set to your hearts content
    def __init__(self, inputnodes,hiddennodes,outputnodes,learningrate):
        # Set number of nodes in each input, hidden, outer layer
        ##Specifies the number of input nodes aka input channels
        self.inodes = inputnodes
        ##Specifies the number of hidden nodes. There is a fixed number of one layer.
        self.hnodes = hiddennodes
        ##Specifies the number of output nodes aka output channels
        self.onodes = outputnodes

        ## Link weight matrix. Weights of connections input ---> hidden
        # weights inside arrays are w_i_j, where link i from node i to j in the next layer
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        ## Link weight matrix. Weights of connections hidden ---> output
        # weights inside arrays are w_i_j, where link i from node i to j in the next layer
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        ## Specifies learning rate
        self.lr = learningrate

        ## Specifies activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    ## Compares current output with the desired output (target) and adjusts the weights accordingly
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = numpy.array(inputs_list,ndmin=2).T 
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #calculate the signals ermerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #calculate error
        output_errors = targets-final_outputs

        #hidden layer error is the output_erros, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    ## Calculates the output of the network, given an input.
    def query(self, inputs_list):
        # convert inputs list to 2D array
        inputs = numpy.array(inputs_list,ndmin=2).T 

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    ## Saves the current trained state of the neural network by writing the weights in .csv files
    # Weights from input to hidden layers are saved in trained_model_wih.csv
    # Weights from hidden to output layers are saved in trained_model_who.csv
    def saveState(self):
        numpy.savetxt('trained_model_wih.csv', self.wih, delimiter=',')
        numpy.savetxt('trained_model_who.csv', self.who, delimiter=',')
    pass

    ## Restores a past trained state of the neural network by reading weights from .csv files
    # Weights from input to hidden layers are read from trained_model_wih.csv
    # Weights from hidden to output layers are read from trained_model_who.csv
    def restoreState(self):
        self.wih = numpy.loadtxt('trained_model_wih.csv', delimiter=',')
        self.who = numpy.loadtxt('trained_model_who.csv', delimiter=',')
    pass
