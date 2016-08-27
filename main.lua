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
require 'image'

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
local Dataset = require("Dataset")
local Model = require("Model")
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

local datasetOpt = {}
datasetOpt.vectorSize = 5
datasetOpt.trainSize = 100
datasetOpt.testSize = 30
datasetOpt.datasetType = 'repeat_k'
datasetOpt.minVal = 1
datasetOpt.maxVal = 31
datasetOpt.memorySize = 5
datasetOpt.repetitions = 2

assert(datasetOpt.maxVal < 2 ^ datasetOpt.vectorSize, "Vector size too small")

local dataset = Dataset.create(datasetOpt)


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a neural architecture to learn algorithms')
cmd:text()
cmd:text('Options')
cmd:option('-trainFile','train.t7', 'filename of the training set')
cmd:option('-testFile', 'test.t7', 'filename of the test set')
cmd:option('-batchSize', 2, 'number of sequences to train in parallel')
cmd:option('-epochs', 30, 'Number of training epochs')

cmd:option('-memorySize', datasetOpt.memorySize,
           'number of entries in linear memory')
cmd:option('-zeroMemory', false, 'Fill unused memory with zeros')
cmd:option('-useCuda', false, 'Should model use cuda')
cmd:option('-noInput', true, 'Architecture used implies separate input')
cmd:option('-maxForwardSteps', '1', 'maximum forward steps model makes')
cmd:option('-saveEvery', 99999, 'save model to file in training after this num')
cmd:option('-saveFile', "autosave.model", 'file to save model in ')
cmd:option('-probabilityDiscount', "0.99", 'probability discount paralel \
    criterion')
cmd:option('-noProb', true, 'Architecture does not emit term. prob.')
cmd:option('-memOnly', true, 'model that uses only memory, no sep input')
cmd:option('supervised' ,true, 'Are we using supervised training')
cmd:option('-eval_episodes', 10, 'Number of evaluation episodes')
cmd:option('-modelName', 'LSTMProb', 'name of model to be used for data dumps')
--------------------------------------------------------------------------------
-- Plotting options
--------------------------------------------------------------------------------
cmd:option('-plot', true, 'Should we plot errors during training')
cmd:option('-plotMemory', true, 'Should we plot memory during training')
cmd:option('-plotAddress', true, 'Should we plot generated addresses')
cmd:option('-plotParams', true, 'Should we plot weights during training')
cmd:option('-sleep', 0, 'Should the beauty sleep?')
--------------------------------------------------------------------------------
-- Hacks section
--------------------------------------------------------------------------------
cmd:option("-giveMeModel", false, "Do not use this at home!")

cmd:text()

local opt = cmd:parse(arg or {})


opt.vectorSize = dataset.vectorSize
opt.inputSize = dataset.inputSize
local ShiftLearn = require('ShiftLearn')

--------------------------------------------------------------------------------
-- TODO should integrate these options nicely
--------------------------------------------------------------------------------

opt.separateValAddr = true 
opt.noInput = true
opt.noProb = false 
opt.simplified = false 
opt.supervised = false 
opt.maxForwardSteps = dataset.repetitions

--------------------------------------------------------------------------------

--local model = Model.create(opt, ShiftLearn.createWrapper,
   --ShiftLearn.createWrapper, nn.Identity, "modelName")

local model = Model.create(opt)

if opt.giveMeModel then
   return model
end

--xavier init
local params, _ = model:parameters()
for k,v in pairs(params) do
    local s = v:size()
    v:apply(function() return torch.normal(0,
        torch.sqrt(3 / s[#s])) end)
end

model.modelName = opt.modelName 

--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

local mse = nn.MSECriterion()
local epochs_all_errors = {}

local epochs_all_errors_mse = {}
local epochs_all_accuracies = {}
local epochs_all_accuracies_mse = {}
local epochs_all_accuracies_discrete = {}
local epochs_all_errors_discrete = {}

opt.simplified = false 
opt.epochs = 5 

local avg_errs = {}
local avg_errs_mse = {}
local avg_errs_discrete = {}

for i=1,opt.epochs do
   model.itNum = i

   local epoch_errors = trainModel(model, nn.PNLLCriterion, dataset, opt, optim.adam)

   local epoch_errors_mse = epoch_errors[1][1]
   local epoch_errors_discrete = epoch_errors[1][2]

   epochs_all_errors_mse[#epochs_all_errors_mse + 1] = unpack(epoch_errors_mse)
   epochs_all_errors_discrete[#epochs_all_errors_discrete + 1] =
      unpack(epoch_errors_discrete) 

   avg_mse = 0.0 
   avg_discrete = 0.0

   for k,v in pairs(epoch_errors_mse) do
      avg_mse = avg_mse + v
   end

   for k,v in pairs(epoch_errors_discrete) do
      avg_discrete = avg_discrete + v
   end

   avg_mse = avg_mse / (#epoch_errors_mse)
   avg_errs_mse[#avg_errs_mse + 1] = avg_mse

   avg_discrete = avg_discrete / (#epoch_errors_discrete)
   avg_errs_discrete[#avg_errs_discrete + 1] = avg_discrete
   local accuracy = {}
   if opt.noProb and opt.supervised then
      accuracy = evalModelSupervised(model, dataset, mse, opt)
   else
      accuracy = evalModelOnDataset(model, dataset, nn.PNLLCriterion, opt)
   end
   epochs_all_accuracies[#epochs_all_accuracies + 1] = accuracy

   epochs_all_accuracies_mse[#epochs_all_accuracies_mse + 1] = accuracy[1]
   epochs_all_accuracies_discrete[#epochs_all_accuracies_discrete + 1] =
      accuracy[2]
  print("end epoch")

end


gnuplot.pngfigure("data_dumps/errors_all_avg_discrete_" .. model.modelName .. 
   "R" .. tostring(dataset.repetitions) .. ".png")
gnuplot.xlabel("Epoch no.")
gnuplot.ylabel("Error(%)")
gnuplot.plot({'Train error',torch.Tensor(avg_errs_discrete)},
   {'Eval error', torch.Tensor(epochs_all_accuracies_discrete)})
gnuplot.plotflush()


gnuplot.pngfigure("data_dumps/errors_all_avg_mse_" .. model.modelName .. 
   "R" .. tostring(dataset.repetitions) .. ".png")
gnuplot.xlabel("Epoch no.")
gnuplot.ylabel("Error")
gnuplot.plot({'Train error', torch.Tensor(avg_errs_mse)},
   {'Eval error', torch.Tensor(epochs_all_accuracies_mse)})
gnuplot.plotflush()
