--------------------------------------------------------------------------------
-- Training unit tests
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
--Training tests definitions
-------------------------------------------------------------------------------
trainingTests = {}


--------------------------------------------------------------------------------
-- Sanity check for training, if this fails, something is wrong at the core of
-- the training procedure used
--------------------------------------------------------------------------------
trainingTests["sanityCheck"] = function()
    Dataset = require("Dataset")
    Model = require("Model")
    require "Training"

    ----------------------------------------------------------------------------
    -- Dummy command line options
    ----------------------------------------------------------------------------
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training a neural architecture to learn algorithms')
    cmd:text()
    cmd:text('Options')
    cmd:option('-trainFile','train.t7', 'filename of the training set')
    cmd:option('-testFile', 'test.t7', 'filename of the test set')
    cmd:option('-batchSize', '16', 'number of sequences to train in parallel')
    cmd:option('-memSize', '20', 'number of entries in linear memory')
    cmd:text()

    local opt = cmd:parse(arg)

    dataset = torch.load(opt.trainFile)
    setmetatable(dataset, Dataset)
    dataset:resetBatchIndex()

    opt.vectorSize = dataset.trainSet[1]:size(3)

    model = Model.create(opt)

   --getting past this point means basic layout of training procedure
    --makes sense
    --trainModel(model, nn.DummyCriterion(), dataset, opt, optim.sgd)
    if pcall(trainModel, model,nn.DummyCriterion(),
            dataset, opt, optim.sgd) then
        return "...OK!"
    end
    return "...fail!"

end
