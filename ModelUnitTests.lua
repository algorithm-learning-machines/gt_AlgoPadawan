--------------------------------------------------------------------------------
-- Model unit tests
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------
require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
require 'optim'
--------------------------------------------------------------------------------
-- Internal modules
--------------------------------------------------------------------------------
Dataset = require("Dataset")
Model = require("Model")


--------------------------------------------------------------------------------
-- Model test definitions
--------------------------------------------------------------------------------
modelTests = {}
modelTests["inputAndMemory"] = function()
   ----------------------------------------------------------------------------
   -- Dummy dataset
   ----------------------------------------------------------------------------
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Generate datasets for learning algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-dataFile','train.t7', 'filename of the training set')
   cmd:option('-vectorSize', 5, 'size of single training instance vector')
   cmd:option('-trainSize', 80, 'size of training set')
   cmd:option('-testSize', 50, 'size of test set')
   cmd:option('-datasetType', 'repeat_binary', 'dataset type')
   cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
   cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
   cmd:option('-memorySize', 500, 'number of entries in memory')
   cmd:text()

   local opt = cmd:parse(arg)

   dataset = Dataset.create(opt)
   ----------------------------------------------------------------------------
   -- Dummy command line options
   ----------------------------------------------------------------------------
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training a neural architecture to learn algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-trainFile','train.t7', 'filename of the training set')
   cmd:option('-testFile', 'test.t7', 'filename of the test set')
   cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
   cmd:text()
   opt = cmd:parse(arg)

   setmetatable(dataset, Dataset)
   dataset:resetBatchIndex()

   opt.vectorSize = dataset.vectorSize
   opt.memorySize = dataset.memorySize
   opt.inputSize = dataset.inputSize

   m = Model.create(opt)
   memory = Tensor(opt.memorySize, opt.vectorSize):fill(0)
   local f = m:forward({memory, dataset:getNextBatch(1)[1][1][1]})

   --check sizes of forward pass to see if they're what we expect
   if f[1]:isSameSizeAs(memory) and f[2]:isSameSizeAs(Tensor(1))
      then
         return "...OK!"
      else
         return "...fail!"
      end

   end

--------------------------------------------------------------------------------
-- Test that model involving only memory works correctly
--------------------------------------------------------------------------------
modelTests["memory"] = function()
   ----------------------------------------------------------------------------
   -- Dummy dataset
   ----------------------------------------------------------------------------
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Generate datasets for learning algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-dataFile','train.t7', 'filename of the training set')
   cmd:option('-vectorSize', 5, 'size of single training instance vector')
   cmd:option('-trainSize', 80, 'size of training set')
   cmd:option('-testSize', 50, 'size of test set')
   cmd:option('-datasetType', 'repeat_binary', 'dataset type')
   cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
   cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
   cmd:option('-memorySize', 500, 'number of entries in memory')
   cmd:text()

   local opt = cmd:parse(arg)

   dataset = Dataset.create(opt)
   ----------------------------------------------------------------------------
   -- Dummy command line options
   ----------------------------------------------------------------------------
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training a neural architecture to learn algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-trainFile','train.t7', 'filename of the training set')
   cmd:option('-testFile', 'test.t7', 'filename of the test set')
   cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
   cmd:text()
   opt = cmd:parse(arg)


   setmetatable(dataset, Dataset)
   dataset:resetBatchIndex()

   opt.vectorSize = dataset.vectorSize
   opt.memorySize = dataset.memorySize
   opt.inputSize = dataset.inputSize
   opt.noInput = true
   m = Model.create(opt)
   memory = Tensor(opt.memorySize, opt.vectorSize):fill(0)
   local f = m:forward(memory)

   --check sizes of forward pass to see if they're what we expect
   if f[1]:isSameSizeAs(memory) and f[2]:isSameSizeAs(Tensor(1))
      then
         return "...OK!"
      else
         return "...fail!"
      end

   end
--------------------------------------------------------------------------------
--modelTests["memory"] = nil

modelTests["saveAndLoad"] = function()
   ----------------------------------------------------------------------------
   -- Dummy dataset
   ----------------------------------------------------------------------------
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Generate datasets for learning algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-dataFile','train.t7', 'filename of the training set')
   cmd:option('-vectorSize', 5, 'size of single training instance vector')
   cmd:option('-trainSize', 80, 'size of training set')
   cmd:option('-testSize', 50, 'size of test set')
   cmd:option('-datasetType', 'repeat_binary', 'dataset type')
   cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
   cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
   cmd:option('-memorySize', 500, 'number of entries in memory')
   cmd:text()

   local opt = cmd:parse(arg)

   dataset = Dataset.create(opt)
   ----------------------------------------------------------------------------
   -- Dummy command line options
   ----------------------------------------------------------------------------
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training a neural architecture to learn algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-trainFile','train.t7', 'filename of the training set')
   cmd:option('-testFile', 'test.t7', 'filename of the test set')
   cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
   cmd:text()
   opt = cmd:parse(arg)

   setmetatable(dataset, Dataset)
   dataset:resetBatchIndex()
   os.execute("rm dummyFile")
   opt.vectorSize = dataset.vectorSize
   opt.memorySize = dataset.memorySize
   opt.inputSize = dataset.inputSize
   opt.noInput = true
   m = Model.create(opt)
   local res = "...OK!"
   local ret = Model.saveModel(m, "dummyFile", false)
   if ret ~= true then
      res = "...fail!"
   end
   ret = Model.saveModel(m, "dummyFile", false)
   if ret ~= false then
      res = "...fail!"
   end
   m = Model.loadModel("dummyFile")
   if m == nil then
      res = "...fail!"
   end
   local mem = torch.zeros(tonumber(opt.memorySize), tonumber(opt.vectorSize))
   
   if not pcall(m.forward, m, mem) then
      res = "...fail!"
   end

  os.execute("rm dummyFile")
  return res

end

--------------------------------------------------------------------------------
-- Test that model involving nram probability model works correctly
--------------------------------------------------------------------------------
modelTests["memoryNRAM"] = function()
   ----------------------------------------------------------------------------
   -- Dummy dataset
   ----------------------------------------------------------------------------
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Generate datasets for learning algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-dataFile','train.t7', 'filename of the training set')
   cmd:option('-vectorSize', 5, 'size of single training instance vector')
   cmd:option('-trainSize', 80, 'size of training set')
   cmd:option('-testSize', 50, 'size of test set')
   cmd:option('-datasetType', 'repeat_binary', 'dataset type')
   cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
   cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
   cmd:option('-memorySize', 500, 'number of entries in memory')
   cmd:option('-NRAMProb', true, 'will we use NRAM probability or not')
   cmd:text()

   local opt = cmd:parse(arg)

   dataset = Dataset.create(opt)
   ----------------------------------------------------------------------------
   -- Dummy command line options
   ----------------------------------------------------------------------------
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training a neural architecture to learn algorithms')
   cmd:text()
   cmd:text('Options')
   cmd:option('-trainFile','train.t7', 'filename of the training set')
   cmd:option('-testFile', 'test.t7', 'filename of the test set')
   cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
   cmd:text()
   opt = cmd:parse(arg)


   setmetatable(dataset, Dataset)
   dataset:resetBatchIndex()

   opt.vectorSize = dataset.vectorSize
   opt.memorySize = dataset.memorySize
   opt.inputSize = dataset.inputSize
   opt.noInput = true -- only memory
   opt.NRAMProb = true --use nram probability model
   m = Model.create(opt)
   memory = Tensor(opt.memorySize, opt.vectorSize):fill(0)
   local f = m:forward({memory, torch.zeros(1)})

   --check sizes of forward pass to see if they're what we expect
   if f[1]:isSameSizeAs(memory) and f[2]:isSameSizeAs(Tensor(1))
      then
         return "...OK!"
      else
         return "...fail!"
      end

   end
--------------------------------------------------------------------------------




