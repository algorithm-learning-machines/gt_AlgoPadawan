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
        --print("final output")
        --print(finalOutput[1])
        --print("label")
        --print(labels[i])

        err = criterion:forward(finalOutput, labels[i])
        errAvg = errAvg + err
    end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    print("Error in evaluation "..errAvg)
end


function getDiffs(outputMem, targetMem, begin_ix, end_ix)
   diff = 0
   for i=begin_ix,end_ix do
      for j=1, targetMem:size(2) do
         local val = outputMem[i][j]
         if val > 0.5 then
            val = 1
         else
            val = 0
         end
         if targetMem[i][j] ~= val then
            diff = diff + 1
         end
      end
   end
   return diff
end

--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset
--------------------------------------------------------------------------------
function evalModelSupervised(model, dataset, criterion, opt)
    local fixedSteps = tonumber(opt.fixedSteps)
    local testSet = dataset.testSet
    local data = testSet[1]
    local labels = testSet[2]
    local errAvg = 0.0
    local prevAdr = torch.zeros(opt.memorySize)
    prevAdr[1] = 1
    for i=1,data:size(1) do
        local currentInstance = data[i]
        local numIterations = 1
        local memory = currentInstance
        local grandErr = 0.0
        for j = 1,opt.maxForwardSteps do
            --print(memory)
            --print(prevAdr)
            output = model:forward({memory, prevAdr})
            --output = model:forward(memory)
            err = criterion:forward(output[1], labels[i][j])
             
            if j == opt.maxForwardSteps then
               err_discrete = getDiffs(output[1], labels[i][j], 1, dataset.repetitions)
               print(labels[i][j])
               print(err_discrete)
            end
            
            grandErr = grandErr + err
            prevAdr = output[2]
            memory = output[1]
        end
        grandErr = grandErr / opt.maxForwardSteps 
        errAvg = errAvg + grandErr
    end

    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    --print("Error in evaluation "..errAvg)
    -- add errors to file
    local file = io.open("data_dumps/errors_evaluation_" .. model.modelName .. tonumber(model.itNum), 'a')
    io.output(file)
    io.write(tostring(errAvg) .. "\n")
    io.output(io.stdout)
    io.close(file)
    return errAvg
end



--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset, supervised, single step
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
end
