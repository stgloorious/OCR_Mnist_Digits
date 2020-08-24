## @package ocr_mnist_digits
#  This module is a demonstration of character recognition based on a neural network
#
#  It uses the MNIST database for the training and testing data. This project 
#  is based upon the book by Tariq Rashid (O'Reilly: "Neurale Netzwerke selbst programmieren").
 
import nn

## CREATE INSTANCE OF NEURAL NETWORK
# input is a 28 by 28 image, 500 hidden nodes. 
# Output is digit indicator 0...9. Learning rate is 0.1.
# Training process is repeated 10 times (epoches)
n = nn.neuralNetwork((28*28),500,10,0.1)

## TRAIN WITH MNIST DATABASE (60'000 samples)
# Training data: http://www.pjreddie.com/media/files/mnist_train.csv
#
nn.train.train(n,"C:/Users/stefa/Desktop/mnist_train.csv",8) # <---- Change this path

## TEST WITH SEPARATE TEST DATA SET (10'000 samples)
# Testing data: http://pjreddie.com/media/files/mnist_test.csv
#
nn.test.test(n,"C:/Users/stefa/Desktop/mnist_test.csv") # <---- Change this path