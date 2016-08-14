--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------
print("gigi are mere")


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
cmd:option('-modelName', 'LSTM', 'name of model to be used for data dumps')
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
opt.noProb = true
opt.simplified = false 
opt.supervised = true 
opt.maxForwardSteps = dataset.repetitions
--------------------------------------------------------------------------------

--local model = Model.create(opt, ShiftLearn.createWrapper,
   --ShiftLearn.createWrapper, nn.Identity, "modelName")

model = Model.create(opt)

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

--------------------------------------------------------------------------------
-- Display parameters before training
--------------------------------------------------------------------------------

--local winsInitial = {}
--local params, _ = model:parameters()
--for k,v in pairs(params) do
   --if v:nDimension() == 1 then
      --winsInitial[k] = image.display{
         --image=v:view(1,-1),
         --win=winsInitial[k],
         --zoom=35,
         --legend = "initial bias " .. k
      --}
   --else
      --winsInitial[k] = image.display{
         --image=v,
         --win=winsInitial[k],
         --zoom=35,
         --legend = "initial params " .. k
      --}
   --end
--end

model.modelName = opt.modelName 
--print(dataset.repetitions)
--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

local mse = nn.MSECriterion()
local epochs_all_errors = {}
local epochs_all_accuracies = {}
opt.simplified = false
opt.epochs = 11 

for i=1,opt.epochs do
   model.itNum = i

   local epoch_errors = trainModel(model, mse, dataset, opt, optim.adam)

   if ((i - 1) % 5 == 0) then
      epochs_all_errors[#epochs_all_errors + 1] = unpack(epoch_errors)
   end

   local accuracy = evalModelSupervised(model, dataset, mse, opt)

   if ((i - 1) % 5 == 0) then
      epochs_all_accuracies[#epochs_all_accuracies + 1] = accuracy
   end

end


local plot_dict = {}

for i=1,#epochs_all_errors do
   ix = 0
   if i == 1 then
      ix = 1
   else
      ix = (i - 1) * 5
   end
   plot_dict[i] = {"epoch num " .. tostring(ix), torch.Tensor(epochs_all_errors[i])}
end

print(plot_dict)

gnuplot.pngfigure("data_dumps/errors_training_" .. model.modelName .. "R" .. tostring(dataset.repetitions) .. ".png")
gnuplot.xlabel("Batch no.")
gnuplot.ylabel("Train Error")
gnuplot.plot(unpack(plot_dict)) -- this has to change
gnuplot.plotflush()

print(epochs_all_accuracies)

gnuplot.pngfigure("data_dumps/errors_eval_" .. model.modelName .. "R" .. tostring(dataset.repetitions) .. ".png")
gnuplot.xlabel("Epoch no.")
gnuplot.ylabel("Eval Error")
gnuplot.plot(torch.Tensor(epochs_all_accuracies)) -- this has to change
gnuplot.plotflush()

--------------------------------------------------------------------------------
-- Evaluate model
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Display parameters after training
--------------------------------------------------------------------------------

--local winsAfter = {}
--local params, _ = model:parameters()
--for k,v in pairs(params) do
   --if v:nDimension() == 1 then
      --winsAfter[k] = image.display{
         --image=v:view(1,-1),
         --win=winsAfter[k],
         --zoom=35,
         --legend = "after bias " .. k
      --}
   --else
      --winsAfter[k] = image.display{
         --image=v,
         --win=winsAfter[k],
         --zoom=35,
         --legend = "after params " .. k
      --}
   --end
--end
