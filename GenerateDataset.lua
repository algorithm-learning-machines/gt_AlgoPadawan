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


--------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------
-- Generate and save data
--------------------------------------------------------------------------------
if (path.exists(opt.dataFile)) then
    print("file "..opt.dataFile.." already exists, please remove it first")
else
    torch.save(opt.dataFile, Dataset.create(opt))
end











