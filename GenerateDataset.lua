--------------------------------------------------------------------------------
-- Contains definitions necessary for generating datasets
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Dependencies
--------------------------------------------------------------------------------
require "torch"

--------------------------------------------------------------------------------
-- Internal modules
--------------------------------------------------------------------------------
Dataset = require("Dataset")
-- TODO temporary hack
Tensor = torch.Tensor

--------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate datasets for learning algorithms')
cmd:text()
cmd:text('Options')
cmd:option('-dataFile','train.t7', 'filename of the training set')
cmd:option('-vectorSize', 10, 'size of single training instance vector')
cmd:option('-trainSize', 100, 'size of training set')
cmd:option('-testSize', 20, 'size of test set')
cmd:option('-datasetType', 'repeat_once', 'dataset type')
cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
cmd:option('-memorySize', 20, 'number of entries in memory')
cmd:option('-maxForwardSteps', '2', 'maximum forward steps model makes')
cmd:text()



local opt = cmd:parse(arg)

--------------------------------------------------------------------------------
-- Generate and save data
--------------------------------------------------------------------------------
if (path.exists(opt.dataFile)) then
    print("file "..opt.dataFile.." already exists, please remove it first")
else
    torch.save(opt.dataFile, Dataset.create(opt))
end











