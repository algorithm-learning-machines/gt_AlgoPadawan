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
trainingTests["TrainOnRepeaterCheck"] = function()
    Dataset = require("Dataset")
    Model = require("Model")
    require "Training"

    ----------------------------------------------------------------------------
    -- Dummy command line options
    ----------------------------------------------------------------------------
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Generate datasets for learning algorithms')
    cmd:text()
    cmd:text('Options')
    cmd:option('-dataFile','train.t7', 'filename of the training set')
    cmd:option('-vectorSize', 5, 'size of single training instance vector')
    cmd:option('-trainSize', 8, 'size of training set')
    cmd:option('-testSize', 5, 'size of test set')
    cmd:option('-datasetType', 'repeat_binary', 'dataset type')
    cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
    cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
    cmd:option('-memorySize', 500, 'number of entries in memory')
    cmd:text()
    local opt = cmd:parse(arg)
    dataset = Dataset.create(opt)
    dataset:resetBatchIndex()



    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Generate datasets for learning algorithms')
    cmd:text()
    cmd:text('Options')
    cmd:option('-trainFile','train.t7', 'filename of the training set')
    cmd:option('-vectorSize', 5, 'size of single training instance vector')
    cmd:option('-trainSize', 3, 'size of training set')
    cmd:option('-testSize', 5, 'size of test set')
    cmd:option('-datasetType', 'repeat_binary', 'dataset type')
    cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
    cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
    cmd:option('-memorySize', 500, 'number of entries in memory')
    cmd:option('-maxForwardSteps', '2', 'maximum forward steps model makes')
    cmd:text()


    local opt = cmd:parse(arg)
    opt.vectorSize = dataset.vectorSize
    opt.memorySize = dataset.memorySize
    opt.inputSize = dataset.inputSize


    model = Model.create(opt)

    opt.batchSize = opt.trainSize
    --getting past this point means basic layout of training procedure
    --makes sense
    --trainModel(model, nn.PNLLCriterion(), dataset, opt, optim.sgd)
    if pcall(trainModel, model,nn.PNLLCriterion(),
        dataset, opt, optim.sgd) then
        return "...OK!"
    end
    return "...fail!"

end

--------------------------------------------------------------------------------
-- Sanity check for training, if this fails, something is wrong at the core of
-- the training procedure used
--------------------------------------------------------------------------------
trainingTests["TrainOnBinaryAdditionCheck"] = function()
    Dataset = require("Dataset")
    Model = require("Model")
    require "Training"

    ----------------------------------------------------------------------------
    -- Dummy command line options
    ----------------------------------------------------------------------------
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Generate datasets for learning algorithms')
    cmd:text()
    cmd:text('Options')
    cmd:option('-trainFile','train.t7', 'filename of the training set')
    cmd:option('-vectorSize', 10, 'size of single training instance vector')
    cmd:option('-trainSize', 3, 'size of training set')
    cmd:option('-testSize', 5, 'size of test set')
    cmd:option('-datasetType', 'binary_addition', 'dataset type')
    cmd:option('-minVal', 1, 'minimum scalar value of dataset instances')
    cmd:option('-maxVal', 300, 'maximum scalar value of dataset instances')
    cmd:option('-memorySize', 500, 'number of entries in memory')
    cmd:option('-maxForwardSteps', '2', 'maximum forward steps model makes')
    cmd:text()


    local opt = cmd:parse(arg)
    dataset = Dataset.create(opt)
    dataset:resetBatchIndex()

    local cmd = torch.Cmd

    opt.vectorSize = dataset.vectorSize
    opt.memorySize = dataset.memorySize
    opt.inputSize = dataset.inputSize
    opt.targetIndex = 1

    model = Model.create(opt)
    opt.batchSize = opt.trainSize
    --getting past this point means basic layout of training procedure
    --makes sense
    --trainModel(model, nn.PNLLCriterion(), dataset, opt, optim.sgd)
    if pcall(trainModel, model,nn.PNLLCriterion(),
        dataset, opt, optim.sgd) then
        return "...OK!"
    end
    return "...fail!"

end

--------------------------------------------------------------------------------
-- Perform gradient checking on custom criterion
--------------------------------------------------------------------------------
trainingTests["criterionChecker"] = function()
    Dataset = require("Dataset")
    Model = require("Model")

    require "Training"
    local criterion = nn.PNLLCriterion()
    i = {}
    i[1] = torch.Tensor(10):fill(0.4)
    i[2] = torch.Tensor(1):fill(0.1)
    local o = torch.Tensor(10):fill(1)
    local r = gradient_check(0.0001, 0.01, criterion, i, o)
    if r == true then
        return "...OK!"
    end
    return "...fail!"

end

--------------------------------------------------------------------------------
-- checks that criterion derivatives are computed correctly
-- h -> addition to gradients used for checking; should be really small
-- e -> tolerance in gradient difference
-- criterion -> the criterion/error function to be minimized
-- input/target -> dummy values to test the gradients
-- returns: true if gradients are computed correctly, false otherwise
--------------------------------------------------------------------------------
function gradient_check(h, e, criterion, input, target)
    ----------------------------------------------------------------------------
    -- Get parameters and gradients now, after execution
    ----------------------------------------------------------------------------
    local err = criterion:forward(input, target)

    ----------------------------------------------------------------------------
    -- Get derivatives with respect to the criterion that we are minimizing
    ----------------------------------------------------------------------------
    local df_do = criterion:backward(input, target)

    ----------------------------------------------------------------------------
    -- Definition of derivative ->  h-> 0; dJ/dW = j(W + h) - j(W - h) / (2*h)
    -- Computed gradient should be similar to this
    ----------------------------------------------------------------------------

    for j=1,#input do
        local params = input[j]
        local dParams = df_do[j]
        -- TODO remove hardcoding
        -- hardcoded for testing
        if j == 2 then
            dParams = torch.Tensor{dParams}
        end

        for i=1,params:size(1) do
            local e_i = torch.zeros(params:size())
            -- phi_i - eps
            params[i] = params[i] - h
            -- J(phi_i - eps)
            local out1 = criterion:forward(input, target)
            -- phi_i + eps
            params[i] = params[i] + 2 * h
            -- J(phi_i + eps)
            local out2 = criterion:forward(input, target)

            local est_g = (out2 - out1) /  (2 * h) -- estimated gradient
            params[i] = params[i] - h -- original param value

            local my_g = dParams[i]
            local rel_err = math.abs(my_g - est_g) / (math.abs(my_g) +
                math.abs(est_g)) -- relative error
            if (rel_err > e) then
                print("param set: "..j)
                print("param_num: "..i)
                print("relative error: "..rel_err)
                print("computed gradient: "..my_g)
                print("estimated gradient: "..est_g)

                return false
            end
        end
    end

    return true
end

