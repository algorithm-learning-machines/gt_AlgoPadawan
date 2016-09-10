--------------------------------------------------------------------------------
-- Definitions referring to the evaluation of a model
--------------------------------------------------------------------------------

-- Needed for criterion
require "Train"
require "gnuplot"
require "image"

--------------------------------------------------------------------------------
-- Evaluate a model trained on a certain dataset
--------------------------------------------------------------------------------
function evalModelOnDataset(model, dataset, criterion, opt)
   local testSet = dataset.testSet
   local data = testSet[1]
   local labels = testSet[2]
   local errAvg = 0.0
   local errAvg_discrete = 0.0
   local prevAdr = torch.zeros(opt.memorySize)
   prevAdr[1] = 1

   for i=1,data:size(1) do

      local currentInstance = data[i]
      local terminated = false
      local numIterations = 0
      local finalOutput = {}

      while not terminated and numIterations < opt.maxForwardSteps do

         local memory = currentInstance
         if opt.simplified then
            output = model:forward({memory, prevAdr})
         else
            output = model:forward(memory)
         end


         local prob = output[#output][1]
         prevAdr = output[2]

         if prob > 0.9 then
            terminated = true
         end

         memory = output[1]
         numIterations = numIterations + 1
         finalOutput = output

      end
      
     
      local comp_index = 0
      local comp_memory = {}
      if dataset.taskName == "repeat_k" then
         comp_index = dataset.repetitions
         comp_memory = labels[i][dataset.repetitions]
      elseif dataset.taskName == "binary_addition" then
         comp_index = 3
         comp_memory = labels[i]
      end
      if opt.simplified or not opt.noProb then
         err_discrete = getDiffs(finalOutput[1], comp_memory, 1,
         comp_index)
      else
         err_discrete = getDiffs(finalOutput, comp_memory, 1,
         comp_index)
      end

      err = criterion:forward(finalOutput, comp_memory)
      errAvg = errAvg + err
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
            print(targetMem[i][j] - val)
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
           if j == opt.maxForwardSteps and i == 1 then
              
              local winsInitial = {}
              local mem_thresh = memory:clone()

              for i=1, dataset.repetitions do
                 for j=1, mem_thresh:size(2) do
                    local val = mem_thresh[i][j]
                    if val > 0.5 then
                       mem_thresh[i][j] = 1
                    else
                       mem_thresh[i][j] = 0
                    end
                 end
              end

              im_output = image.display{
                 image = mem_thresh,
                 win = im_output,
                 zoom=70,
                 legend = "output_memory" 
              }
              im_input = image.display{
                 image = data[i],
                 win = im_input,
                 zoom=70,
                 legend = "input memory" 
              }
              im_target = image.display{
                 image = labels[i][j],
                 win = im_target,
                 zoom=70,
                 legend = "target" 
              }
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
