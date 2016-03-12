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
modelTests["sanityCheck"] = function()
    ----------------------------------------------------------------------------
    -- Dummy dataset
    ----------------------------------------------------------------------------
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Generate datasets for learning algorithms')
    cmd:text()
    cmd:text('Options')
    cmd:option('-dataFile','train.t7', 'filename of the training set')
    cmd:option('-vectorSize', 20, 'size of single training instance vector')
    cmd:option('-trainSize', 1000, 'size of training set')
    cmd:option('-testSize', 50, 'size of test set')
    cmd:option('-datasetType', 'binary_addition', 'dataset type')
    cmd:option('-minVal', 0, 'minimum scalar value of dataset instances')
    cmd:option('-maxVal', 5000,'maximum scalar value of dataset instances')
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
    cmd:option('-memSize', '20', 'number of entries in linear memory')
    cmd:text()

    opt = cmd:parse(arg)

    ----------------------------------------------------------------------------
    -- Run sanity check on size
    ----------------------------------------------------------------------------

    setmetatable(dataset, Dataset)
    dataset:resetBatchIndex()


    opt.vectorSize = dataset.trainSet[1]:size(3)

    m = Model.create(opt)
    memory = torch.Tensor(tonumber(opt.memSize), opt.vectorSize):fill(0)

    local f = m:forward({memory, dataset:getNextBatch(1)[1][1][1]})

    --check sizes of forward pass to see if they're what we expect
    if f[1]:isSameSizeAs(memory) and f[2]:isSameSizeAs(torch.Tensor(1))
    then
        return "...OK!"
    else
        return "...fail!"
    end

end
--------------------------------------------------------------------------------
