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

    for i=self.batchIndex,math.min(self.batchIndex + batchSize - 1, self.trainSize) do
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

    if opt.datasetType == "binary_addition" then

        local trainSet, trainNumbers = Dataset.__genBinaryAdditionSet(self.trainSize,
            self.vectorSize, self.minVal, self.maxVal, {})
        local testSet, _ = Dataset.__genBinaryAdditionSet(self.testSize,
            self.vectorSize, self.minVal, self.maxVal, trainNumbers)

        self.trainSet = trainSet
        self.testSet = testSet

    else
        print("Dataset type " .. opt.dataset_type .. "Not implemented yet!")
        os.exit()
    end

    return self
end


--------------------------------------------------------------------------------
-- generate a binary addition set
-- setSize -> size of set
-- range -> minimum and maximum values
-- vectorSize -> size of the binary vector; should be log2(maxNum)
--------------------------------------------------------------------------------
function Dataset.__genBinaryAdditionSet(setSize, vectorSize, minVal,
    maxVal, exclusionSet)
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
        local c = a + b
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

