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

    local batch = {torch.Tensor(batchSize, self.trainSet[1]:size(2),
        self.trainSet[1]:size(3)), torch.Tensor(batchSize,
        self.trainSet[2]:size(2), self.trainSet[2]:size(3))}

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
--------------------------------------------------------------------------------
function Dataset.create(opt)
    self = {}
    setmetatable(self, Dataset)

    self.trainSize  = tonumber(opt.trainSize)
    self.testSize   = tonumber(opt.testSize)
    self.vectorSize = tonumber(opt.vectorSize)
    self.minVal = tonumber(opt.minVal)
    self.maxVal = tonumber(opt.maxVal)
    self.batchIndex = 1 -- initial index
    
    local trainSet = {}
    local testSet = {}
    if opt.datasetType == "binary_addition" then
        local trainNumbers = {}
         trainSet, trainNumbers = Dataset.__genBinaryOpSet(
            self.trainSize, self.vectorSize, self.minVal, self.maxVal, {},
            function(a,b) return a + b end)
         testSet, _ = Dataset.__genBinaryOpSet(self.testSize,
            self.vectorSize, self.minVal, self.maxVal, trainNumbers,
            function(a,b) return a + b end)

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
    else
        print("Dataset type " .. opt.dataset_type .. "Not implemented yet!")
        os.exit()
    end

    self.trainSet = trainSet
    self.testSet = testSet

    return self
end


--------------------------------------------------------------------------------
-- generate a binary operation set
-- setSize -> size of set
-- range -> minimum and maximum values
-- vectorSize -> size of the binary vector; should be log2(maxNum)
-- f -> function that gets executed on input
--------------------------------------------------------------------------------
function Dataset.__genBinaryOpSet(setSize, vectorSize, minVal,
    maxVal, exclusionSet, f)
    local input = torch.Tensor(setSize,  2, vectorSize)
    local target = torch.Tensor(setSize, 1, vectorSize)
    local inputOriginal = {}
    for i=1,setSize do
        local a = math.random(minVal, maxVal)
        local b = math.random(minVal, maxVal)
        while (inputOriginal[tostring(a).."_"..tostring(b)] ~= nil) or
            (exclusionSet[tostring(a).."_"..tostring(b)] ~= nil) do
            a = math.random(minVal, maxVal)
            b = math.random(minVal, maxVal)
        end
        inputOriginal[tostring(a).."_"..tostring(b)] = {a, b}
        local c = f(a,b)
        local aVec =Dataset.__numToBits(a, vectorSize)
        local bVec =Dataset.__numToBits(b, vectorSize)
        local cVec =Dataset.__numToBits(c, vectorSize)
        local entry = torch.cat(aVec,bVec,2)
        entry = entry:t()
        input[i] = entry
        target[i] = cVec
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
    local input = torch.Tensor(setSize,  2, vectorSize)
    local target = torch.Tensor(setSize, 1, vectorSize)
    local inputOriginal = {}
    for i=1,setSize do
        local a = torch.random(torch.Tensor(vectorSize), minVal, maxVal)
        local b = torch.random(torch.Tensor(vectorSize), minVal, maxVal)
        while (inputOriginal[tostring(a).."_"..tostring(b)] ~= nil) or
            (exclusionSet[tostring(a).."_"..tostring(b)] ~= nil) do
            a = torch.random(torch.Tensor(vectorSize), minVal, maxVal)
            b = torch.random(torch.Tensor(vectorSize), minVal, maxVal)
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



--------------------------------------------------------------------------------
-- convert number to bit vector
-- num -> number to convert
-- bits -> number of bits that shall represent the number
--------------------------------------------------------------------------------
function Dataset.__numToBits(num, bits)
    local bitVec = torch.Tensor(bits, 1):fill(0)
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

