Graduation Thesis -> Adrian-Mihai Iosif

An implementation of an architecture capable of learning algorithms


Dependencies:
    - torch
    - nn
    - nngraph
    - dpnn
    - rnn -> Recurrent Library for Torch. arXiv preprint arXiv:1511.07889 (2015)

File Description:
main.lua
        -> main entry point of application
Dataset.lua
        -> dataset specific classes and functions
GenerateDataset.lua
        -> generates a specific dataset according to given params
Model.lua
        -> model definition
ModelUnitTests.lua
        -> unit tests referring to model definitions
DataUnitTests.lua
        -> unit tests referring to dataset definitions
Training.lua
        -> contains training  classes and functions, such as custom Criterions



