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
cmd:option('-maxForwardSteps', '10', 'maximum forward steps model makes')
cmd:option('-saveEvery', 1, 'save model to file in training after this num')
cmd:option('-saveFile', "autosave.model", 'file to save model in ')
cmd:text()

local opt = cmd:parse(arg)

if opt.useCuda then
    Tensor = torch.CudaTensor
else
    Tensor = torch.Tensor
end


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
runUnitTestSuite(modelTests)
print("Running data tests...")
runUnitTestSuite(dataTests)
--print("Running training tests...")
--runUnitTestSuite(trainingTests)

--------------------------------------------------------------------------------
-- Train on repeat-once dataset
--------------------------------------------------------------------------------

local dataset = torch.load(opt.trainFile)
setmetatable(dataset, Dataset)

opt.vectorSize = dataset.vectorSize
opt.inputSize = dataset.inputSize

local model = Model.create(opt)
--TODO hack, should integrate this in model somewhere
--model.maxForwardSteps = 15
trainModelNoMemory(model,nn.PNLLCriterion(), dataset, opt, optim.sgd)
--model = torch.load("autosave.model")
evalModelOnDataset(model, dataset, nn.PNLLCriterion())


