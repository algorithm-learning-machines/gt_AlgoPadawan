--------------------------------------------------------------------------------
-- File containing Training definitions, for example Criterions,
-- Custom optimizing procedures
--------------------------------------------------------------------------------



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

--------------------------------------------------------------------------------
-- Returns loss function
--------------------------------------------------------------------------------
function PNLLCriterion:forward(input, target)
    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    local prob = input[2][1]
    return (-1) * prob * self:sumDifference(input, target)
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
    local f1 = memory:clone():log():cmul(target)  --tk * log(mk)
    local f2a = torch.ones(memSize) - target
    local f2b = torch.ones(memSize) - torch.log(memory) --(1 - tk) * log(1 - mk)
    local f2 = f2a:cmul(f2b) -- tk * log(mk) + (1 - tk) * log(1 - mk)
    return f2:sum()

end

--------------------------------------------------------------------------------
-- Derivatives of relevant memory and probabiltiy
--------------------------------------------------------------------------------
function PNLLCriterion:backward(input, target)
    ----------------------------------------------------------------------------
    -- Extract info from parameters
    ----------------------------------------------------------------------------
    local memory = input[1]
    local memSize = memory:size()
    local prob = input[2][1]

    ----------------------------------------------------------------------------
    -- Derivative of probabilty
    ----------------------------------------------------------------------------
    local dProb = self:sumDifference(input, target)

    ----------------------------------------------------------------------------
    -- Derivative of memory
    ----------------------------------------------------------------------------
    local dMemory = torch.Tensor(memSize):fill(prob)
    local denom = memory + target - torch.ones(memSize)
    dMemory:cdiv(denom)
    dMemory = dMemory * (-1)

    self.gradInput = {dMemory, dProb}
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
    parameters, gradParameters = model:getParameters()

    model:training()
    batch = dataset:getNextBatch(tonumber(opt.batchSize))
    
    while batch ~= nil do

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = 1, opt.batchSize do
            -- load new sample
            local input = batch[1][i]
            local target = batch[2][i]
            table.insert(inputs, input)
            table.insert(targets, target)
        end

        batch = dataset:getNextBatch(tonumber(opt.batchSize))
        batch = nil -- force rapid exit for now
        -- create closure to evaluate f(X) and df/dX
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
                -- dummy memory for now
                local memSize = tonumber(opt.memSize)
                local memory =
                    torch.Tensor(memSize, opt.vectorSize):fill(0)
                -- TODO forward until prob is 1 or threshhold steps
                -- TODO propagate gradients backwards

                local output = model:forward({memory, inputs[i][1]})
                -- Loss is only interested in first row
                output[1] = output[1][{1}]

                local err = criterion:forward(output, targets[i])
                f = f + err

                -- estimate df/dW
                local df_do = criterion:backward(output, targets[i])
                -- consider derivatives for 'scratchpad' mem are 0
                local memoryDev = torch.cat(df_do[1]:reshape(1,df_do[1]:size(1))
                    , torch.zeros(memSize-1, opt.vectorSize), 1)

                df_do[1] = memoryDev
                df_do[2] = torch.Tensor{df_do[2]}

                --will need both memory and probability derivatives
                model:backward({memory, inputs[i][1]}, df_do)
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
    end
end



