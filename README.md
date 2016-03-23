##Graduation Thesis -> Adrian-Mihai Iosif##

An implementation of an architecture capable of learning algorithms


##Dependencies:##
* torch
* nn
* nngraph
* dpnn
* rnn -> Recurrent Library for Torch. arXiv preprint arXiv:1511.07889 (2015)

##File Description:##
###main.lua###
Main entry point of application. Should run GenerateDataset script first.
###Dataset.lua###
Dataset specific classes and functions
###DataUnitTests.lua###
Unit tests referring to dataset definitions
###GenerateDataset.lua###
Generates a specific dataset according to given params
###Model.lua###
Model definition
###ModelUnitTests.lua###
Unit tests referring to model definitions
###Training.lua###
Contains training classes and functions, such as custom Criterions
###TrainingUnitTests.lua###
Unit tests referring to training definitions


