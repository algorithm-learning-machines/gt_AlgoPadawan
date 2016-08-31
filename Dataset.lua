--------------------------------------------------------------------------------
-- Dataset related functions and classes
--------------------------------------------------------------------------------

local Dataset = {}
Dataset.__index = Dataset

--------------------------------------------------------------------------------
-- Returns next batch
-- modifies current batchIndex
--------------------------------------------------------------------------------
function Dataset:getNextBatch(batchSize)
   if (self.batchIndex + batchSize - 1 > self.trainSize) then
      return nil
   end
   local args_input = {}
   local args_lable = {}
   args_input[1] = batchSize
   args_lable[1] = batchSize
   for i = 2,#self.trainSet[1]:size() do
      args_input[#args_input + 1] = self.trainSet[1]:size(i)
   end

   for i = 2,#self.trainSet[2]:size() do
      args_lable[#args_lable + 1] = self.trainSet[2]:size(i)
   end

   local batch = {Tensor(unpack(args_input)), Tensor(unpack(args_lable))}

   for i=self.batchIndex,math.min(self.batchIndex + batchSize - 1,
      self.trainSize) do
      batch[1][i - self.batchIndex + 1] = self.trainSet[1][i]
      batch[2][i - self.batchIndex + 1] = self.trainSet[2][i]
   end

   self.batchIndex = self.batchIndex + batchSize

   return batch

end



--------------------------------------------------------------------------------
-- Resets the batch index
--------------------------------------------------------------------------------
function Dataset:resetBatchIndex()
   self.batchIndex = 1
end

--------------------------------------------------------------------------------
-- Init dataset according to given parameters from main entry point
-- Dataset shape differs
--------------------------------------------------------------------------------
function Dataset.create(opt)

   self = {}
   setmetatable(self, Dataset)

   self.trainSize  = tonumber(opt.trainSize)
   self.testSize   = tonumber(opt.testSize)
   self.vectorSize = tonumber(opt.vectorSize)
   self.minVal = tonumber(opt.minVal)
   self.maxVal = tonumber(opt.maxVal)
   self.memorySize = tonumber(opt.memorySize)
   self.batchIndex = 1 -- initial index
   self.taskName = opt.datasetType

   local trainSet = {}
   local testSet = {}

   if opt.datasetType == "binary_addition" then
      self.inputSize = self.vectorSize
      local trainNumbers = {}
      trainSet, trainNumbers = Dataset.__genBinaryOpSet(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
      function(a,b) return a + b end, self.memorySize)
      testSet, _ = Dataset.__genBinaryOpSet(self.testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers,
      function(a,b) return a + b end, self.memorySize)

   elseif opt.datasetType == "binary_cmmdc" then
      local trainNumbers = {}
      trainSet, trainNumbers = Dataset.__genBinaryOpSet(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
      cmmdc)
      testSet, _ = Dataset.__genBinaryOpSet(self.testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers,
      cmmdc)

   elseif opt.datasetType == "addition" then
      local trainNumbers = {}
      trainSet, trainNumbers = Dataset.__genAdditionSet(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {})
      testSet, _ = Dataset.__genAdditionSet(self. testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers)

   elseif opt.datasetType == "repeat_binary" then
      self.inputSize = 1 -- specifically for this dataset
      trainSet, trainNumbers = Dataset.__genRepeatSet(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
      self.memorySize)
      testSet, _ = Dataset.__genRepeatSet(self. testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers,
      self.memorySize)

   elseif opt.datasetType == "repeat_once" then
      self.inputSize = self.vectorSize
      trainSet, trainNumbers = Dataset.__genRepeatOnce(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
      self.memorySize)
      testSet, _ = Dataset.__genRepeatOnce(self. testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers,
      self.memorySize)

   elseif opt.datasetType == "repeat_k" then
      self.repetitions = tonumber(opt.repetitions)
      self.inputSize = self.vectorSize
      trainSet, trainNumbers = Dataset.__genRepeatK(
      self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
      self.memorySize, self.repetitions)
      testSet, _ = Dataset.__genRepeatK(self. testSize,
      self.vectorSize, self.minVal, self.maxVal, trainNumbers,
      self.memorySize, self.repetitions)
   else
      print("Dataset type " .. opt.dataset_type .. "Not implemented yet!")
      os.exit()
   end

   self.trainSet = trainSet
   self.testSet = testSet

   return self
end

--------------------------------------------------------------------------------
-- Creates dataset with the following form:
-- input: Initial Memory (Template on first position)
-- target: Final Memory (Template on first two positions)
--------------------------------------------------------------------------------
function Dataset.__genRepeatOnce(
   setSize, vectorSize, minVal, maxVal, exclusionSet, memorySize)
   local inputs = Tensor(setSize, memorySize, vectorSize)
   local labels = Tensor(setSize, memorySize, vectorSize)


   local inputOriginal = {}
   for i=1,setSize do
      local template = Dataset.__numToBits(torch.random(minVal, maxVal),
      vectorSize)
      while (inputOriginal[tostring(template)] ~= nil) or
         (exclusionSet[tostring(template)] ~= nil) do
         template = Dataset.__numToBits(torch.random(minVal, maxVal),
         vectorSize)
      end

      local inputMemory = torch.repeatTensor(torch.zeros(1, vectorSize),
      memorySize - 1, 1)
      inputMemory = torch.cat(template:t(), inputMemory, 1)
      inputs[i] = inputMemory
      local idx = torch.linspace(1,inputs[i]:size(1), inputs[i]:size(1)):long()
      idx[1], idx[2] = 2, 1
      inputs[i] = inputs[i]:index(1, idx)

      inputOriginal[tostring(template)] = template
      outputMemory = inputMemory:clone()
      outputMemory[2] = template:clone()
      labels[i] = outputMemory
   end
   return {inputs, labels}, inputOriginal

end


--------------------------------------------------------------------------------
-- Creates dataset with the following form:
-- scalar -> scalar * vector-of-ones
-- Task would be to repeat the same pattern of ones for a specific num of times
--------------------------------------------------------------------------------
function Dataset.__genRepeatSet(setSize, vectorSize, minVal, maxVal,
   exclusionSet, memorySize)

   local input = Tensor(setSize, 1, 1)
   local labels = Tensor(setSize, memorySize, vectorSize)
   local pattern = torch.random(torch.Tensor(vectorSize), 0, 1)


   local inputOriginal = {}
   for i=1,setSize do
      local times = torch.random(minVal, maxVal)
      while (inputOriginal[tostring(times)] ~= nil) or
         (exclusionSet[tostring(times)] ~= nil) do
         times = torch.random(minVal, maxVal)
      end
      input[i][1] = torch.Tensor{times}
      inputOriginal[tostring(times)] = times

      ------------------------------------------------------------------------
      -- repeat the pattern specified times
      -- fill rest of memory with zero
      ------------------------------------------------------------------------
      local filler = torch.zeros(memorySize - times, vectorSize)
      local repeatedPattern = torch.repeatTensor(pattern, times, 1)
      local target = torch.cat(repeatedPattern, filler, 1)
      labels[i] = target
   end
   return {input, labels}, inputOriginal
end


--------------------------------------------------------------------------------
-- generate a binary operation set
-- setSize -> size of set
-- range -> minimum and maximum values
-- vectorSize -> size of the binary vector; should be log2(maxNum)
-- f -> function that gets executed on input
--------------------------------------------------------------------------------
function Dataset.__genBinaryOpSet(setSize, vectorSize, minVal,
   maxVal, exclusionSet, f, memorySize)
   local input = Tensor(setSize,  memorySize, vectorSize)
   local target = Tensor(setSize, memorySize, vectorSize)
   local inputOriginal = {}

   for i=1,setSize do
      local a = math.random(minVal, maxVal / 2)
      local b = math.random(minVal, maxVal / 2)
      while (inputOriginal[tostring(a).."_"..tostring(b)] ~= nil) or
         (exclusionSet[tostring(a).."_"..tostring(b)] ~= nil) do
         a = math.random(minVal, maxVal / 2)
         b = math.random(minVal, maxVal / 2)
      end
      inputOriginal[tostring(a).."_"..tostring(b)] = {a, b}
      local c = f(a,b)
      local aVec = Dataset.__numToBits(a, vectorSize)
      local bVec = Dataset.__numToBits(b, vectorSize)
      local cVec = Dataset.__numToBits(c, vectorSize)

      local entry = torch.cat(aVec, bVec, 2)
      local randomPart = torch.rand(memorySize, vectorSize)
      entry = entry:t()

      local inputFiller = randomPart[{{3, memorySize}}]
      local targetFiller = randomPart[{{4, memorySize}}]

      local outputMemory = nil

      outputMemory = torch.cat(entry, cVec:t(), 1)
      outputMemory = torch.cat(outputMemory, targetFiller, 1)
      
      input[i] = torch.cat(entry, inputFiller, 1) 
      target[i] = outputMemory 
   end
   return {input, target}, inputOriginal
end


--------------------------------------------------------------------------------
--Greatest common divisor of a and b
--------------------------------------------------------------------------------
function cmmdc(a, b)
   while b ~= 0 do
      r = a % b
      a = b
      b = r
   end
   return a
end


--------------------------------------------------------------------------------
-- generate an addition set
-- setSize -> size of set
-- range -> minimum and maximum values
-- vectorSize -> size of the vector -> initially thought to be 1
--------------------------------------------------------------------------------
function Dataset.__genAdditionSet(setSize, vectorSize, minVal,
   maxVal, exclusionSet)
   local input = Tensor(setSize,  2, vectorSize)
   local target = Tensor(setSize, 1, vectorSize)
   local inputOriginal = {}
   for i=1,setSize do
      local a = torch.random(Tensor(vectorSize), minVal, maxVal)
      local b = torch.random(Tensor(vectorSize), minVal, maxVal)
      while (inputOriginal[tostring(a).."_"..tostring(b)] ~= nil) or
         (exclusionSet[tostring(a).."_"..tostring(b)] ~= nil) do
         a = torch.random(Tensor(vectorSize), minVal, maxVal)
         b = torch.random(Tensor(vectorSize), minVal, maxVal)
      end
      inputOriginal[tostring(a).."_"..tostring(b)] = {a, b}
      local c = a + b
      local entry = torch.cat(a,b,2)
      entry = entry:t()
      input[i] = entry
      target[i] = c
   end
   return {input, target}, inputOriginal
end

function Dataset.__genRepeatK(setSize, vectorSize,
   minVal, maxVal, exclusionSet, memorySize, repetitions)

   --repetitions no of memories
   local inputs = Tensor(setSize, memorySize, vectorSize)
   local labels = Tensor(setSize, repetitions, memorySize, vectorSize)

   local inputOriginal = {}
   for i=1,setSize do
      local template = Dataset.__numToBits(torch.random(minVal, maxVal),
      vectorSize)

      --[[
      while (inputOriginal[tostring(template)] ~= nil) or
         (exclusionSet[tostring(template)] ~= nil) do
         template = Dataset.__numToBits(torch.random(minVal, maxVal),
         vectorSize)
      end
      --]]

      inputOriginal[tostring(template)] = template

      --local randomPart = torch.repeatTensor(torch.rand(1, vectorSize),
         --memorySize - 1, 1)

      local randomPart = torch.rand(memorySize - 1, vectorSize)
      --torch.rand(memorySize - 1, vectorSize)

      inputMemory = torch.cat(template:t(), randomPart, 1)
      inputs[i] = inputMemory

      for j=1,repetitions do
         local outputRepeated = torch.repeatTensor(template:t(), j + 1, 1)
         local zeroed

         if j ~= memorySize then
            --zeroed = torch.repeatTensor(torch.zeros(1, vectorSize),
            --memorySize - j - 1, 1)
            --zeroed = torch.Tensor(memorySize - j - 1, vectorSize)
            --print(randomPart)
            zeroed = randomPart[{{j + 1, randomPart:size(1)}}]
         end

         local outputMemory = nil
         if zeroed ~= nil then
            outputMemory = torch.cat(outputRepeated, zeroed, 1)
         else
            outputMemory = outputRepeated
         end

         labels[i][j] = outputMemory
      end
   end

   return {inputs, labels}, inputOriginal

end


--------------------------------------------------------------------------------
-- convert number to bit vector
-- num -> number to convert
-- bits -> number of bits that shall represent the number
--------------------------------------------------------------------------------
function Dataset.__numToBits(num, bits)
   local bitVec = Tensor(bits, 1):fill(0)
   local i_bit = 1
   while num ~= 0 do
      local b = bit.band(num, 1)
      bitVec[i_bit][1] = b
      num = bit.rshift(num, 1)
      i_bit = i_bit + 1
   end
   return bitVec
end

return Dataset
