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
require 'cutorch'


--------------------------------------------------------------------------------
-- Command line options 
--------------------------------------------------------------------------------
Tensor = torch.DoubleTensor
--------------------------------------------------------------------------------
-- Internal modules
--------------------------------------------------------------------------------
require 'ModelUnitTests'
require 'DataUnitTests'
require 'TrainingUnitTests'
Dataset = require("Dataset")
Model = require("Model")
require "Train"
require "Evaluation"

--------------------------------------------------------------------------------
-- Run unit Tests
--------------------------------------------------------------------------------
function runUnitTestSuite(testSuite)
    for k,v in pairs(testSuite) do
        print(k..v())
    end
end

print("Running model tests...")
--runUnitTestSuite(modelTests)
print("Running data tests...")
--runUnitTestSuite(dataTests)
print("Running training tests...")
--runUnitTestSuite(trainingTests)

--------------------------------------------------------------------------------
-- Train on repeat-once dataset
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
-- Create dataset
--------------------------------------------------------------------------------

opt = {}
opt.vectorSize = 15
opt.trainSize = 1000
opt.testSize = 30
opt.datasetType = 'repeat_k'
opt.minVal = 1
opt.maxVal = 5000
opt.memorySize = 20
opt.repetitions = 1
local dataset = Dataset.create(opt) 
print(dataset.trainSet[1][2])



local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a neural architecture to learn algorithms')
cmd:text()
cmd:text('Options')
cmd:option('-trainFile','train.t7', 'filename of the training set')
cmd:option('-testFile', 'test.t7', 'filename of the test set')
cmd:option('-batchSize', '1', 'number of sequences to train in parallel')
cmd:option('-memorySize', '20', 'number of entries in linear memory')
cmd:option('-useCuda', false, 'Should model use cuda')
cmd:option('-noInput', true, 'Architecture used implies separate input')
cmd:option('-maxForwardSteps', '1', 'maximum forward steps model makes')
cmd:option('-saveEvery', 5, 'save model to file in training after this num')
cmd:option('-saveFile', "autosave.model", 'file to save model in ')
cmd:option('-probabilityDiscount', "0.99", 'probability discount paralel \
    criterion')
cmd:option('-noProb', true, 'Architecture does not emit term. prob.')
cmd:option('-memOnly', true, 'model that uses only memory, no sep input')
cmd:option('supervised' ,true, 'Are we using supervised training')
cmd:option('-plot', true, 'Should we plot errors during training')
cmd:text()

local opt = cmd:parse(arg)




--setmetatable(dataset, Dataset)

opt.vectorSize = dataset.vectorSize
opt.inputSize = dataset.inputSize
local ShiftLearn = require('ShiftLearn')

--------------------------------------------------------------------------------
-- TODO should integrate these options nicely
--------------------------------------------------------------------------------
opt.separateValAddr = true
opt.noInput = true 
opt.noProb = true
opt.simplified = true
opt.supervised = true
opt.maxForwardSteps = dataset.repetitions
--------------------------------------------------------------------------------

local model = Model.create(opt, ShiftLearn.createWrapper,
   ShiftLearn.createWrapper, nn.Identity)

----xavier init
local params, _ = model:parameters()
for k,v in pairs(params) do
    local s = v:size()
    v:apply(function() return torch.normal(0,
        torch.sqrt(3 / s[#s])) end)
end

local mse = nn.MSECriterion()

trainModel(model, mse, dataset, opt, optim.adam)
--evalModelSupervised(model, dataset, mse, opt)


