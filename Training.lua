--------------------------------------------------------------------------------
-- File containing Training definitions, for example Criterions,
-- Custom optimizing procedures
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
--Dummy Criterion for prototyping
--------------------------------------------------------------------------------
local DummyCriterion, _ = torch.class('nn.DummyCriterion',  'nn.Criterion')

function DummyCriterion:forward(input, target)
    return 0
end

function DummyCriterion:backward(input, target)
    self.gradInput = input:clone():fill(0)
    return self.gradInput
end


--------------------------------------------------------------------------------
-- function that trains a model on a dataset using a certain criterion and
-- optimization method
-- opt represents table with command line parameters received from main
-- entry point of application
-- Current state represents just a sketch of the final trainig procedure
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
                local memory =
                    torch.Tensor(tonumber(opt.memSize), opt.vectorSize):fill(0)

                local output = model:forward({memory, inputs[i][1]})
                local err = criterion:forward(output, targets[i])
                f = f + err

                -- estimate df/dW
                local df_do = criterion:backward(output[1], targets[i])
                --will need both memory and probability derivatives
                model:backward({memory, inputs[i][1]}, {df_do, torch.Tensor(1)})

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



