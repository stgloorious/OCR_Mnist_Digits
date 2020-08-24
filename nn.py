import numpy
import scipy.special

class neuralNetwork:
    def __init__(self, inputnodes,hiddennodes,outputnodes,learningrate):
        # Set number of nodes in each input, hidden, outer layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Link weight matrices, wih and who
        # weights inside arrays are w_i_j, where link i from node i to j in the next layer
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

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