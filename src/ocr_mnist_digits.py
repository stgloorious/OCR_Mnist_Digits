## @package ocr_mnist_digits
#  This module is a demonstration of character recognition based on a neural network
#
#  It uses the MNIST database for the training and testing data. This project 
#  is based upon the book by Tariq Rashid (O'Reilly: "Neurale Netzwerke selbst programmieren").
 
import nn

## CREATE INSTANCE OF NEURAL NETWORK
# input is a 28 by 28 image, 500 hidden nodes. 
# Output is digit indicator 0...9. Learning rate is 0.1.
n = nn.neuralNetwork((28*28),1000,10,0.1)

## TRAIN WITH MNIST DATABASE (60'000 samples)
# Training data: http://www.pjreddie.com/media/files/mnist_train.csv
#
# Training process is repeated 5 times (epoches)
# Training does not have to be done every time, since the resulting weights are saved in a .csv
# file and automatically restored before testing
#
# vvv Uncomment next line to train neural network vvv
#
nn.train.train(n,"C:/Users/stefa/Dropbox/Documents/Hobby/Software/Python/DigitDatasetCreator/merged_training.csv",5) # <---- Change this path

## TEST WITH SEPARATE TEST DATA SET (10'000 samples)
# Testing data: http://pjreddie.com/media/files/mnist_test.csv
#
#
nn.test.test(n,"C:/Users/stefa/Dropbox/Documents/Hobby/Software/Python/OCR_Mnist_Digits/data/mnist_test.csv") # <---- Change this path

nn.test.test(n,"C:/Users/stefa/Dropbox/Documents/Hobby/Software/Python/DigitDatasetCreator/data/testing.csv") # <---- Change this path

