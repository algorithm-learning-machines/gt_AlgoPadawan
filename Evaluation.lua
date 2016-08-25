--------------------------------------------------------------------------------
-- Definitions referring to the evaluation of a model
--------------------------------------------------------------------------------

-- Needed for criterion
require "Train"
require "gnuplot"

--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset
--------------------------------------------------------------------------------
function evalModelOnDataset(model, dataset, criterion, opt)
    local testSet = dataset.testSet
    local data = testSet[1]
    local labels = testSet[2]
    local errAvg = 0.0
    local errAvg_discrete = 0.0

    for i=1,data:size(1) do

        local currentInstance = data[i]
        local terminated = false
        local numIterations = 0
        local finalOutput = {}

        while not terminated and numIterations < opt.maxForwardSteps do

            local memory = currentInstance
            local output = model:forward(currentInstance)

            local prob = output[2][1]

            if prob > 0.9 then
                terminated = true
            end
            print(prob)
            memory = output[1]
            numIterations = numIterations + 1
            finalOutput = output

        end

        err = criterion:forward(finalOutput, labels[i][dataset.repetitions])
        errAvg = errAvg + err
        
        if opt.simplified or not opt.noProb then
           err_discrete = getDiffs(finalOutput[1], labels[i][dataset.repetitions], 1,
           dataset.repetitions)
        else
           err_discrete = getDiffs(finalOutput, labels[i][dataset.repetitions], 1,
            dataset.repetitions)
        end

        errAvg_discrete = errAvg_discrete + err_discrete 


     end
    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    errAvg_discrete = errAvg_discrete / data:size(1)
    print("Error in evaluation "..errAvg)
    return {errAvg, errAvg_discrete}
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
   return diff * 1.0 / ((end_ix - begin_ix + 1) * targetMem:size(2)) * 100.0
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
    local errAvg_discrete = 0.0
    local prevAdr = torch.zeros(opt.memorySize)
    local err_discrete = 0.0
    prevAdr[1] = 1
    local grandErr = 0.0
    local grandErr_discrete = 0.0
    for i=1,data:size(1) do
        local currentInstance = data[i]
        local numIterations = 1
        local memory = currentInstance

        for j = 1,opt.maxForwardSteps do
           if opt.simplified then
              output = model:forward({memory, prevAdr})
              err = criterion:forward(output[1], labels[i][j])

           else
              output = model:forward(memory)
              err = criterion:forward(output, labels[i][j])
           end
            
            --output = model:forward(memory)
                         
            if j == opt.maxForwardSteps then
               if opt.simplified then
                  err_discrete = getDiffs(output[1], labels[i][j], 1,
                     dataset.repetitions)
               else
                  err_discrete = getDiffs(output, labels[i][j], 1,
                     dataset.repetitions)
               end
            end
            
            grandErr = grandErr + err
            if opt.simplified then
               prevAdr = output[2]
               memory = output[1]
            else 
               memory = output
            end
         end
        grandErr = grandErr / opt.maxForwardSteps 
         
        errAvg = errAvg + grandErr
        errAvg_discrete = errAvg_discrete + err_discrete 
    end

    ----------------------------------------------------------------------------
    -- Average error
    ----------------------------------------------------------------------------
    errAvg = errAvg / data:size(1)
    errAvg_discrete = errAvg_discrete / data:size(1)
    
    return {errAvg, errAvg_discrete}
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
