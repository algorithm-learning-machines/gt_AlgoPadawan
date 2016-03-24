--------------------------------------------------------------------------------
-- Definitions referring to the evaluation of a model
--------------------------------------------------------------------------------

-- Needed for criterion
require "Training"
require "gnuplot"

--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset
--------------------------------------------------------------------------------
function evalModelOnDataset(model, dataset, criterion)
    local testSet = dataset.testSet
    local data = testSet[1]
    local labels = testSet[2]
    local errAvg = 0.0
    for i=1,data:size(1) do
        local currentInstance = data[i]
        local terminated = false
        local numIterations = 0
        local finalOutput = {}
        while not terminated and numIterations < model.maxForwardSteps do
            local memory = currentInstance
            local output = model:forward(currentInstance)
            local prob = output[2][1]
            if prob > 0.9 then
                terminated = true
            end
            memory = output[1]
            numIterations = numIterations + 1
            finalOutput = output
        end
        err = criterion:forward(finalOutput, labels[i])
        errAvg = errAvg + err
    end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    print("Error in evaluation "..errAvg)
end
