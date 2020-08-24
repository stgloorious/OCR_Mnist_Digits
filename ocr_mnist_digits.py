## @package ocr_mnist_digits
#  This module is a demonstration of character recognition based on a neural network
#
#  It uses the MNIST database for the training and testing data. This project 
#  is based upon the book by Tariq Rashid (O'Reilly: "Neurale Netzwerke selbst programmieren").
 
import nn

## CREATE INSTANCE OF NEURAL NETWORK
n = nn.neuralNetwork((28*28),100,10,0.3)
# input is a 28 by 28 image, 100 hidden layers. 
# output are digit indicators 0...9. Learning rate is 0.3.

## TRAIN WITH MNIST DATABASE (60'000 samples)
nn.train.train(n,"C:/Users/stefa/Desktop/mnist_train.csv")

## TEST WITH SEPARATE TEST DATA SET (10'000 samples)
nn.test.test(n,"C:/Users/stefa/Desktop/mnist_test.csv")