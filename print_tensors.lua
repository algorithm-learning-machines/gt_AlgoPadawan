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


local winInput
local winTarget
local winsParams = {}
local winsGradParams = {}
local winsParamsBias = {}
local winsGradParamsBias = {}
local winsAddress = {}

--image.display(inp)



local datasetOpt = {}
datasetOpt.vectorSize = 5
datasetOpt.trainSize = 100
datasetOpt.testSize = 30
datasetOpt.datasetType = 'binary_addition'
datasetOpt.minVal = 1
datasetOpt.maxVal = 31 
datasetOpt.memorySize = 5
datasetOpt.repetitions = 3 
datasetOpt.maxForwardSteps = 5

local dataset = Dataset.create(datasetOpt)


dataset:resetBatchIndex()
batchSize = 1
batch = dataset:getNextBatch(batchSize)

local input = batch[1][1]
local target = batch[2][1]
--print(input)
--print(target)

local inp1 = image.toDisplayTensor(input,2,2)
local inp2 = image.toDisplayTensor(target,2,2)

winInput1 = image.display{
   image=inp1, offscreen=false,  
   zoom=100, legend='produced memory'
}

winInput2 = image.display{
   image=inp2, offscreen=false,  
   zoom=100, legend='produced memory'
}
----winInpu
----print(winInput)


--image.save("gigi.png",inp)
