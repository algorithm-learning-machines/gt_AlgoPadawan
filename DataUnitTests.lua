--------------------------------------------------------------------------------
-- File containing unit tests for generated datasets
--------------------------------------------------------------------------------

require "torch"
require "nn"
require "nngraph"
require "rnn"
require "cunn"
local Dataset = require("Dataset")


dataTests = {}

--------------------------------------------------------------------------------
-- Test batch system
--------------------------------------------------------------------------------
dataTests["testBatch"] =  function()
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

    new_dataset = Dataset.create(opt)

    new_dataset:getNextBatch(10)
    local count = 0
    local batch = new_dataset:getNextBatch(10)
    while batch ~= nil do
        count = count + batch[1]:size(1)
        batch = new_dataset:getNextBatch(10)
    end

    if count ~= 990 then
        return "...failed!"
    end
    return  "...OK!"
end

--------------------------------------------------------------------------------
-- Test function that transforms scalar to binary vector
--------------------------------------------------------------------------------
dataTests["testScalarToBinary"] = function()
    ----------------------------------------------------------------------------
    -- Dummy options
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

    local testA = Tensor(5):fill(0) -- 3
    local genA = Dataset.__numToBits(3,5)
    testA[1] = 1
    testA[2] = 1
    testA:resize(5,1)
    if testA:eq(genA):all() then
        return "...OK!"
    else
        return "...failed!"
    end
end







