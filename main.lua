--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------
require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
require 'optim'
require 'ModelUnitTests'
require 'DataUnitTests'
require 'TrainingUnitTests'


--------------------------------------------------------------------------------
-- Internal modules
--------------------------------------------------------------------------------
Dataset = require("Dataset")
Model = require("Model")
require "Training"

--------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a neural architecture to learn algorithms')
cmd:text()
cmd:text('Options')
cmd:option('-trainFile','train.t7', 'filename of the training set')
cmd:option('-testFile', 'test.t7', 'filename of the test set')
cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
cmd:option('-memSize', '20', 'number of entries in linear memory')
cmd:text()

local opt = cmd:parse(arg)

--------------------------------------------------------------------------------
-- Run unit Tests
--------------------------------------------------------------------------------
function runUnitTestSuite(testSuite)
    for k,v in pairs(testSuite) do
        print(k..v())
    end
end

print("Running model tests...")
runUnitTestSuite(modelTests)
print("Running data tests...")
runUnitTestSuite(dataTests)
print("Running training tests...")
runUnitTestSuite(trainingTests)




