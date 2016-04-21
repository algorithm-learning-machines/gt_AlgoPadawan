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
-- Command line options --------------------------------------------------------------------------------

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
--runUnitTestSuite(modelTests)
print("Running data tests...")
--runUnitTestSuite(dataTests)
print("Running training tests...")
--runUnitTestSuite(trainingTests)

--------------------------------------------------------------------------------
-- Train on repeat-once dataset
--------------------------------------------------------------------------------

local dataset = torch.load(opt.trainFile)
setmetatable(dataset, Dataset)

opt.vectorSize = dataset.vectorSize
opt.inputSize = dataset.inputSize
local ShiftLearn = require('ShiftLearn')

local gigi = ShiftLearn.create(10)
--print(gigi:forward({torch.zeros(10), torch.zeros(10)}))
local p1 = ShiftLearn.createWrapper(tonumber(opt.memorySize))

--print(p1:forward(torch.zeros(opt.memorySize)))
--print(p1)
--print(ShiftLearn.createWrapper(tonumber(opt.memorySize))())
opt.separateValAddr = true
opt.noInput = false

local model = Model.create(opt)
--model:forward(
   --{torch.zeros(tonumber(opt.memorySize),tonumber(opt.vectorSize)),
   --torch.zeros(tonumber(opt.vectorSize))})
   

----xavier init
--local params, _ = model:parameters()
--for k,v in pairs(params) do
    --local s = v:size()
    --v:apply(function() return torch.normal(0,
        --torch.sqrt(3 / s[#s])) end)
--end

--local mse = nn.MSECriterion()


--trainModel(model, mse, dataset, opt, optim.adam)
--evalModelSupervised(model, dataset, mse, opt)


