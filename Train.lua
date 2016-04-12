--------------------------------------------------------------------------------
-- File containing Training definitions, for example Criterions,
-- Custom optimizing procedures
--------------------------------------------------------------------------------
require 'gnuplot'


--------------------------------------------------------------------------------
-- Dummy Criterion for prototyping
--------------------------------------------------------------------------------
local DummyCriterion, _ = torch.class('nn.DummyCriterion',  'nn.Criterion')

function DummyCriterion:forward(input, target)
    return 0
end

function DummyCriterion:backward(input, target)
    self.gradInput = torch.zeros(input:size())
    return self.gradInput
end



--------------------------------------------------------------------------------
-- Probability negative log likelihood
-- Custom defined criterion
--------------------------------------------------------------------------------
local PNLLCriterion, _ = torch.class('nn.PNLLCriterion',  'nn.Criterion')


function PNLLCriterion:updateOutput(input, target)
    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    local prob = input[2][1]
    self.output = (-1) * prob * self:sumDifference(input, target)
    return self.output
end


--------------------------------------------------------------------------------
-- sum over k of tk * log(mk) + (1-tk) * log(1 - mk)
--------------------------------------------------------------------------------
function PNLLCriterion:sumDifference(input, target)
    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    local memory = input[1]
    local memSize = memory:size()

    ----------------------------------------------------------------------------
    -- Vectorize loss calculus
    ----------------------------------------------------------------------------
    local f1 = memory:clone():log():cmul(target)  -- tk * log(mk)
    local f2a = torch.ones(memSize) - target
    local f2b = torch.log(torch.ones(memSize) - memory) --(1-tk) * log(1-mk)
    local f2 = f2a:cmul(f2b) -- tk * log(mk) + (1 - tk) * log(1 - mk)
    return (f1 + f2):sum()

end


function PNLLCriterion:updateGradInput(input, target)
  ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    local memory = input[1]
    local memSize = memory:size()
    local prob = input[2][1]

    ----------------------------------------------------------------------------
    -- Derivative of probabilty
    ----------------------------------------------------------------------------
    local dProb = (-1) * self:sumDifference(input, target)

    ----------------------------------------------------------------------------
    -- Derivative of memory
    ----------------------------------------------------------------------------
    local dMemory = Tensor(memSize):fill(prob)
    local denom = memory + target - torch.ones(memSize)
    dMemory:cdiv(denom)
    dMemory = dMemory * (-1)
    self.gradInput = {dMemory, torch.Tensor{dProb}}
    return self.gradInput

end


--------------------------------------------------------------------------------
-- function that trains a model on a dataset using a certain criterion and
-- optimization method
-- opt represents table with command line parameters received from main
-- entry point of application
-- Current state represents just a sketch of the final training procedure
-- Heavily inspired from torch tutorials on https://github.com/torch/tutorials/
--------------------------------------------------------------------------------
function trainModel(model, criterion, dataset, opt, optimMethod)

    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    parameters, gradParameters = model:getParameters()
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    local batchSize = tonumber(opt.batchSize)
    local maxForwardSteps = tonumber(opt.maxForwardSteps)
    ----------------------------------------------------------------------------
    -- Work in batches
    ----------------------------------------------------------------------------
    --model:training() -- set model in training mode

    batch = dataset:getNextBatch(batchSize)
    local batchNum = 1
    ----------------------------------------------------------------------------
    -- Training loop
    ----------------------------------------------------------------------------
    while batch ~= nil do
        --gradParameters:zero()
        batchNum = batchNum + 1
        ------------------------------------------------------------------------
        -- Create mini batches
        ------------------------------------------------------------------------
        local inputs = {}
        local targets = {}
        for i = 1, batchSize do
            -- load new sample
            local input = batch[1][i]
            local target = batch[2][i]
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        batch = dataset:getNextBatch(batchSize)
        ------------------------------------------------------------------------
        -- Create closure to evaluate f(X) and df/dX
        ------------------------------------------------------------------------
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local memory = Tensor(memSize, vectorSize):fill(0)
                ----------------------------------------------------------------
                -- Forward until probability comes close to 1 or until max
                -- number of forwards steps has been reached
                ----------------------------------------------------------------
                local terminated = false
                local numIterations = 0
                local clones = {}
                local cloneInputs = {}
                local cloneOutputs = {}
                local probabilities = {}
                clones[0] = model
                local inputsIndex = 1 -- current input index;
                while (not terminated) and numIterations < maxForwardSteps do
                    local currentInput = nil
                    if inputsIndex <= inputs[i]:size(1) then
                        currentInput = inputs[i][inputsIndex]
                    else
                        currentInput = torch.zeros(inputs[i][1]:size())
                    end
                    cloneInputs[numIterations] = {memory, currentInput}

                    local output = clones[numIterations]:forward(
                        cloneInputs[numIterations])


                    cloneOutputs[numIterations] = output -- needed for Criterion
                    probabilitties[numIterations] = output[2]
                    print(probabilities[numIterations])
                    numIterations = numIterations + 1

                    ------------------------------------------------------------
                    -- Remember models and their respective inputs
                    ------------------------------------------------------------

                    clones[numIterations] = cloneModel(model) -- clone model

                    -- needed for backprop
                    memory = output[1]
                    inputsIndex = inputsIndex + 1
                end

                ----------------------------------------------------------------
                -- Propagate gradients from front to back; cumulate gradients
                ----------------------------------------------------------------

                local err = 0
                for j=#clones - 1,0,-1 do

                    local currentOutput = cloneOutputs[j]
                    if opt.targetIndex ~= nil then
                        local ix = tonumber(opt.targetIndex)
                        currentOutput[1] =
                            currentOutput[1][{{1, ix}, {}}]:t():squeeze()
                    end
                    ------------------------------------------------------------
                    -- Find error and output gradients at this time step
                    ------------------------------------------------------------
                    local prob_target = torch.Tensor{0}
                    if j > 1 then
                        prob_target = torch.Tensor{1}
                    end
                    local currentErr = criterion:forward({currentOutput[2],
                        currentOutput},
                    {prob_target, targets[i]})
                    local currentDf_do = criterion:backward(currentOutput,
                        targets[i])

                    if opt.targetIndex ~= nil then
                        local memoryDev = torch.cat(currentDf_do[1]:reshape(1,
                        currentDf_do[1]:size(1)),
                        torch.zeros(memSize-1, opt.vectorSize), 1)
                        currentDf_do[1] = memoryDev
                    end

                    ------------------------------------------------------------
                    -- Output derivatives
                    ------------------------------------------------------------
                    currentDf_do[2] = Tensor{currentDf_do[2]}

                    clones[j]:backward(cloneInputs[i],
                        currentDf_do)

                    err = err + currentErr
                end
                f = f + err
                collectgarbage()
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs

            -- return f and df/dX
            return f,gradParameters
        end

        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        collectgarbage()
    end
end

--------------------------------------------------------------------------------
-- Clone a neural net model
-- Heavily inspired from:
-- https://github.com/oxford-cs-ml-2015/practical6/blob/master/model_utils.lua
--------------------------------------------------------------------------------
function cloneModel(model)
    local params, gradParams
    if model.parameters then
        params, gradParams = model:parameters()
        if params == nil then
            params = {}
        end
    end
    local paramsNoGrad
    if model.parametersNoGrad then
        paramsNoGrad = model:parametersNoGrad()
    end
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(model)

    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if model.parameters then
        local cloneParams, cloneGradParams = clone:parameters()
        local cloneParamsNoGrad
        for i = 1, #params do
             --Sets reference to model's parameters
             cloneParams[i]:set(params[i])
             cloneGradParams[i]:set(gradParams[i])

       end
        if paramsNoGrad then
            cloneParamsNoGrad = clone:parametersNoGrad()
            for i =1,#paramsNoGrad do
                ---- Sets reference to model's parameters
                cloneParamsNoGrad[i]:set(paramsNoGrad[i])
            end
        end
    end
    collectgarbage()
    mem:close()
    return clone
end


--------------------------------------------------------------------------------
-- function that trains a model on a dataset using a certain criterion and
-- optimization method
-- opt represents table with command line parameters received from main
-- entry point of application
-- Current state represents just a sketch of the final training procedure
-- Heavily inspired from torch tutorials on https://github.com/torch/tutorials/
--------------------------------------------------------------------------------
function trainModelOnlyMem(model, criterion, dataset, opt, optimMethod)

    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    parameters, gradParameters = model:getParameters()
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    local batchSize = tonumber(opt.batchSize)
    local maxForwardSteps = tonumber(opt.maxForwardSteps)
    local saveEvery = tonumber(opt.saveEvery)
    ----------------------------------------------------------------------------
    -- Work in batches
    ----------------------------------------------------------------------------
    --model:training() -- set model in training mode

    batch = dataset:getNextBatch(batchSize)
    local batchNum = 1
    local errors = {}
    local learnIterations = 0
    ----------------------------------------------------------------------------
    -- Training loop
    ----------------------------------------------------------------------------
    while batch ~= nil do
        learnIterations = learnIterations + 1
        batchNum = batchNum + 1
        local timer = torch.Timer()
        ------------------------------------------------------------------------
        -- Create mini batches
        ------------------------------------------------------------------------
        local inputs = {}
        local targets = {}
        for i = 1, batchSize do
            -- load new sample
            local input = batch[1][i]
            local target = batch[2][i]
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        batch = dataset:getNextBatch(batchSize)
        ------------------------------------------------------------------------
        -- Create closure to evaluate f(X) and df/dX
        ------------------------------------------------------------------------
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local memory = inputs[i]
                if (not opt.memOnly == true) then
                    memory = Tensor(memSize, vectorSize):fill(0)
                end

                ----------------------------------------------------------------
                -- Forward until probability comes close to 1 or until max
                -- number of forwards steps has been reached
                ----------------------------------------------------------------
                local terminated = false
                local numIterations = 0
                local clones = {}
                local cloneInputs = {}
                local cloneOutputs = {}
                clones[0] = model
                local inputsIndex = 1 -- current input index;
                while (not terminated) and numIterations < maxForwardSteps do
                    local currentInput = nil
                    if (not opt.memOnly == true) then
                        if inputsIndex <= inputs[i]:size(1) then
                            currentInput = inputs[i][inputsIndex]
                        else
                            currentInput = torch.zeros(inputs[i][1]:size())
                        end
                        cloneInputs[numIterations] = {memory, currentInput}
                    else
                        cloneInputs[numIterations] = memory
                    end
                    local output = clones[numIterations]:forward(
                        cloneInputs[numIterations])

                    local prob_val = output[2][1]
                    print(prob_val)
                    cloneOutputs[numIterations] = output -- needed for Criterion

                    numIterations = numIterations + 1
                    ------------------------------------------------------------
                    -- Remember models and their respective inputs
                    ------------------------------------------------------------

                    clones[numIterations] = cloneModel(model) -- clone model
                    -- needed for backprop

                    memory = output[1]
                    inputsIndex = inputsIndex + 1
                    if (prob_val > 0.9) then
                        terminated = true
                    end
                end
                ----------------------------------------------------------------
                -- Propagate gradients from front to back; cumulate gradients
                ----------------------------------------------------------------
                local err = 0
                for j=#clones - 1,0,-1 do

                    local currentOutput = cloneOutputs[j]
                    if opt.targetIndex ~= nil then
                        local ix = tonumber(opt.targetIndex)
                        currentOutput[1] =
                            currentOutput[1][{{1, ix}, {}}]:t():squeeze()
                    end

                    ------------------------------------------------------------
                    -- Find error and output gradients at this time step
                    ------------------------------------------------------------

                    ------------------------------------------------------------
                    -- Target probability given to auxiliary MSE Criterion
                    ------------------------------------------------------------
                    local prob_target = torch.Tensor{0}
                    if j > 1 then
                        prob_target = torch.Tensor{1}
                    end
                    local currentErr = criterion:forward(
                        {currentOutput[2]:clone(), currentOutput},
                    {prob_target, targets[i]})
                    ------------------------------------------------------------

                    local currentDf_do = criterion:backward(
                        {currentOutput[2], currentOutput},
                        {prob_target, targets[i]})

                    if opt.targetIndex ~= nil then
                        local memoryDev = torch.cat(currentDf_do[1]:reshape(1,
                        currentDf_do[1]:size(1)),
                        torch.zeros(memSize-1, opt.vectorSize), 1)
                        currentDf_do[1] = memoryDev
                    end

                    ------------------------------------------------------------
                    -- Output derivatives
                    ------------------------------------------------------------
                    --add probability errors to initial criterion
                    currentDf_do[2][2]:add(currentDf_do[1])
                    ------------------------------------------------------------
                    clones[j]:backward(cloneInputs[i],
                        currentDf_do[2])

                    err = err + currentErr
                end
                -- Temporary checker
                f = f + err
                collectgarbage()
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs
            errors[#errors + 1]  = f
            --------------------------------------------------------------------
            -- Intermediary plot
            --------------------------------------------------------------------
            gnuplot.plot(torch.Tensor(errors))
            -- return f and df/dX
            return f,gradParameters
        end
        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        print("time to process batch.."..timer:time().real..' seconds')
        timer:reset()
        ------------------------------------------------------------------------
        -- Save model to file
        ------------------------------------------------------------------------
        if saveEvery ~= nil and learnIterations % saveEvery == 0 then
            local ret = Model.saveModel(opt.saveFile)
            if ret ~= true then
                print("Model saving could not be finalized")
                error({code=121})
            else
                print("Model has been saved to "..opt.saveFile)
            end
        end
        print("error "..errors[#errors])
    end
    ----------------------------------------------------------------------------
    -- Plot errors for reference
    ---------------------------------------------------------------------------
    gnuplot.plot(torch.Tensor(errors))
end


--------------------------------------------------------------------------------
-- function that trains a model on a dataset using a certain criterion and
-- optimization method
-- opt represents table with command line parameters received from main
-- entry point of application
-- Current state represents just a sketch of the final training procedure
-- Heavily inspired from torch tutorials on https://github.com/torch/tutorials/
--------------------------------------------------------------------------------
function trainModelNoInputOrProb(model, criterion, dataset, opt, optimMethod)

    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    parameters, gradParameters = model:getParameters()
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    local batchSize = tonumber(opt.batchSize)
    local maxForwardSteps = tonumber(opt.maxForwardSteps)
    local saveEvery = tonumber(opt.saveEvery)
    ----------------------------------------------------------------------------
    -- Work in batches
    ----------------------------------------------------------------------------
    model:training() -- set model in training mode
    --print(parameters)
    batch = dataset:getNextBatch(batchSize)
    local batchNum = 1
    local errors = {}
    local learnIterations = 0
    ----------------------------------------------------------------------------
    -- Training loop
    ----------------------------------------------------------------------------
    while batch ~= nil do
        learnIterations = learnIterations + 1
        batchNum = batchNum + 1
        local timer = torch.Timer()
        ------------------------------------------------------------------------
        -- Create mini batches
        ------------------------------------------------------------------------
        local inputs = {}
        local targets = {}
        for i = 1, batchSize do
            -- load new sample
            local input = batch[1][i]
            local target = batch[2][i]
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        batch = dataset:getNextBatch(batchSize)
        ------------------------------------------------------------------------
        -- Create closure to evaluate f(X) and df/dX
        ------------------------------------------------------------------------

        --print(inputs[1])
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local memory = inputs[i]
                print(memory:size())
                local output = model:forward(memory)
                local y_size, x_size = output:size(1), output:size(2)
                local currentErr = criterion:forward(output, targets[i])
                local currentDf_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], currentDf_do)

                -- Temporary checker
                f = f + currentErr
                collectgarbage()
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs
            errors[#errors + 1]  = f
            --------------------------------------------------------------------
            -- Intermediary plot
            --------------------------------------------------------------------
            gnuplot.plot(torch.Tensor(errors))
            -- return f and df/dX
            return f, gradParameters
        end
        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        print("time to process batch.."..timer:time().real..' seconds')
        timer:reset()
        ------------------------------------------------------------------------
        -- Save model to file
        ------------------------------------------------------------------------
        if saveEvery ~= nil and learnIterations % saveEvery == 0 then
            local ret = Model.saveModel(opt.saveFile)
            if ret ~= true then
                print("Model saving could not be finalized")
                error({code=121})
            else
                print("Model has been saved to "..opt.saveFile)
            end
        end
        print("error "..errors[#errors])
    end
    ----------------------------------------------------------------------------
    -- Plot errors for reference
    ---------------------------------------------------------------------------
    gnuplot.plot(torch.Tensor(errors))
end

--------------------------------------------------------------------------------
-- function that trains a model on through step by step supervised learning
-- Heavily inspired from torch tutorials on https://github.com/torch/tutorials/
--------------------------------------------------------------------------------
function trainModelSupervisedSteps(model, criterion, dataset, opt, optimMethod)

    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    parameters, gradParameters = model:getParameters()
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    local batchSize = tonumber(opt.batchSize)
    local saveEvery = tonumber(opt.saveEvery)
    local fixedSteps = tonumber(opt.fixedSteps)
    ----------------------------------------------------------------------------
    -- Work in batches
    ----------------------------------------------------------------------------
    --model:training() -- set model in training mode

    batch = dataset:getNextBatchSupervisedSBS(batchSize)
    local batchNum = 1
    local errors = {}
    local learnIterations = 0
    ----------------------------------------------------------------------------
    -- Training loop
    ----------------------------------------------------------------------------
    while batch ~= nil do
        learnIterations = learnIterations + 1
        batchNum = batchNum + 1
        local timer = torch.Timer()
        ------------------------------------------------------------------------
        -- Create mini batches
        ------------------------------------------------------------------------
        local inputs = {}
        local targets = {}

        for i = 1, batchSize do
            -- load new sample
            local input = batch[1][i]
            local target = batch[2][i]
            table.insert(inputs, input)
            table.insert(targets, target)
        end

        batch = dataset:getNextBatchSupervisedSBS(batchSize)
        ------------------------------------------------------------------------
        -- Create closure to evaluate f(X) and df/dX
        ------------------------------------------------------------------------
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local memory = inputs[i]
                if (not opt.memOnly == true) then
                    memory = Tensor(memSize, vectorSize):fill(0)
                end

                ----------------------------------------------------------------
                -- Forward until probability comes close to 1 or until max
                -- number of forwards steps has been reached
                ----------------------------------------------------------------
                local terminated = false
                local numIterations = 1
                local clones = {}
                local cloneInputs = {}
                local cloneOutputs = {}
                clones[1] = model
                local inputsIndex = 1 --current input index;
                while numIterations <= fixedSteps do
                    local currentInput = nil
                    if (not opt.memOnly == true) then
                        if inputsIndex <= inputs[i]:size(1) then
                            currentInput = inputs[i][inputsIndex]
                        else
                            currentInput = torch.zeros(inputs[i][1]:size())
                        end
                        cloneInputs[numIterations] = {memory, currentInput}
                    else
                        cloneInputs[numIterations] = memory
                    end
                    local output = clones[numIterations]:forward(
                        cloneInputs[numIterations])

                    cloneOutputs[numIterations] = output -- needed for Criterion

                    numIterations = numIterations + 1
                    ------------------------------------------------------------
                    -- Remember models and their respective inputs
                    ------------------------------------------------------------
                    clones[numIterations] = cloneModel(model) -- clone model

                    -- needed for backprop
                    memory = output
                    inputsIndex = inputsIndex + 1
                end
                ----------------------------------------------------------------
                -- Propagate gradients from front to back; cumulate gradients
                ----------------------------------------------------------------
                print(#clones)
                local err = 0
                for j=#clones - 1,1,-1 do

                    local currentOutput = cloneOutputs[j]
                    if opt.targetIndex ~= nil then
                        local ix = tonumber(opt.targetIndex)
                        currentOutput[1] =
                            currentOutput[1][{{1, ix}, {}}]:t():squeeze()
                    end

                    ------------------------------------------------------------
                    -- Find error and output gradients at this time step
                    ------------------------------------------------------------
                    print("----------------------------------------------stuff")
                    print(currentOutput)
                    print(targets[i][j])
                    print("----------------------------------------other stuff")
                    local currentErr = criterion:forward(
                        currentOutput, targets[i][j])
                    ------------------------------------------------------------

                    local currentDf_do = criterion:backward(
                        currentOutput, targets[i][j]) --step by step learning

                    if opt.targetIndex ~= nil then
                        local memoryDev = torch.cat(currentDf_do[1]:reshape(1,
                        currentDf_do[1]:size(1)),
                        torch.zeros(memSize-1, opt.vectorSize), 1)
                        currentDf_do[1] = memoryDev
                    end

                    clones[j]:backward(cloneInputs[i], currentDf_do)

                    err = err + currentErr
                end
                -- Temporary checker
                f = f + err
                collectgarbage()
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs
            errors[#errors + 1]  = f
            --------------------------------------------------------------------
            -- Intermediary plot
            --------------------------------------------------------------------
            gnuplot.plot(torch.Tensor(errors))
            -- return f and df/dX
            return f,gradParameters
        end
        -- optimize on current mini-batch
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
        print("time to process batch.."..timer:time().real..' seconds')
        timer:reset()
        ------------------------------------------------------------------------
        -- Save model to file
        ------------------------------------------------------------------------
        if saveEvery ~= nil and learnIterations % saveEvery == 0 then
            local ret = Model.saveModel(opt.saveFile)
            if ret ~= true then
                print("Model saving could not be finalized")
                error({code=121})
            else
                print("Model has been saved to "..opt.saveFile)
            end
        end
        print("error "..errors[#errors])
    end
    ----------------------------------------------------------------------------
    -- Plot errors for reference
    ---------------------------------------------------------------------------
    gnuplot.plot(torch.Tensor(errors))
end




