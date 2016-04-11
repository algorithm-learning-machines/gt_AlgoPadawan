--------------------------------------------------------------------------------
-- Definitions referring to the evaluation of a model
--------------------------------------------------------------------------------

-- Needed for criterion
require "Train"
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
        print("final output")
        print(finalOutput[1])
        print("label")
        print(labels[i])

        err = criterion:forward(finalOutput, labels[i])
        errAvg = errAvg + err
    end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    print("Error in evaluation "..errAvg)
end


--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset
--------------------------------------------------------------------------------
function evalModelSupervisedSteps(model, dataset, criterion, opt)
    local fixedSteps = tonumber(opt.fixedSteps)
    local testSet = dataset.testSet
    local data = testSet[1]
    local labels = testSet[2]
    --print(labels)
    local errAvg = 0.0
    for i=1,data:size(1) do
        local currentInstance = data[i]
        local numIterations = 1
        local memory = currentInstance
        local grandErr = 0.0
        for j = 1,fixedSteps do
            output = model:forward(memory)
            print(output:size())
            print(labels[i]:size())
            err = criterion:forward(output, labels[i][j])
            grandErr = grandErr + err
            print("LABEL-------------")
            print(labels[i][j])
            print("OUTPUT------------")
            print(output)
            print("END---------------")
            memory = output
        end
        grandErr = grandErr / fixedSteps
        local output = model:forward(currentInstance)
        errAvg = errAvg + grandErr
    end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    print("Error in evaluation "..errAvg)
end



--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset, trained
--------------------------------------------------------------------------------
function evalModelOnDatasetNoProb(model, dataset, criterion)
    local testSet = dataset.testSet
    local data = testSet[1]
    local labels = testSet[2]
    local errAvg = 0.0
    for i=1,data:size(1) do
        local currentInstance = data[i]
        local terminated = false
        local numIterations = 0
        local memory = currentInstance
        local output = model:forward(currentInstance)
        print("LABEL-------------")
        print(labels[i])
        print("OUTPUT------------")
        print(output)
        print("END---------------")
        err = criterion:forward(output, labels[i])
        errAvg = errAvg + err
    end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    print("Error in evaluation "..errAvg)
end
