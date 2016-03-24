--------------------------------------------------------------------------------
-- File containing model definition
--------------------------------------------------------------------------------
require "rnn" require "nngraph"
--require "cutorch"
--require "cunn"
local Model = {}


--------------------------------------------------------------------------------
-- Creates model, may use external input besides
--------------------------------------------------------------------------------
function Model.create(opt)
    if opt.noInput == true then
        return Model.__createNoInput(opt)
    end
    return Model.__createWithInput(opt)
end

--------------------------------------------------------------------------------
-- Creates model with separate input and memory
--------------------------------------------------------------------------------
function Model.__createWithInput(opt)
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    local inputSize = tonumber(opt.inputSize)

    ----------------------------------------------------------------------------
    --  Initial Memory
    ----------------------------------------------------------------------------
    local initialMem = nn.Identity()()
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- Input
    ----------------------------------------------------------------------------
    local input = nn.Identity()()
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    --  Address Encoder
    ----------------------------------------------------------------------------
    local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
    local enc = nn.GRU(memSize * vectorSize, memSize)(reshapedMem)
    local address = nn.SoftMax()(enc)
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    -- Value Extractor
    ----------------------------------------------------------------------------
    
    local addressTransp = nn.Reshape(1, memSize)(address)
    local value = nn.MM()({addressTransp, initialMem})
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- Next address calculator
    ----------------------------------------------------------------------------
    local reshapedValue = nn.Squeeze(1)(value)
    local inputVal = nn.JoinTable(1)({input, address})
    local inputValAddr = nn.JoinTable(1)({inputVal, reshapedValue})
    local addrCalc =
        nn.GRU(inputSize + memSize + vectorSize, memSize)(inputValAddr)
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    -- Next value calculator
    ----------------------------------------------------------------------------
    local valueCalc =
        nn.GRU(inputSize + memSize + vectorSize, vectorSize)(inputValAddr)
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- Memory Calculator
    ----------------------------------------------------------------------------

    --adder
    local resizeValueCalc = nn.Reshape(1, vectorSize)(valueCalc)
    local resizeAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
    local adder = nn.MM()({resizeAddrCalc, resizeValueCalc})


    -- eraser
    local addrCalcTransp = nn.Reshape(1, memSize)(addrCalc)
    local AT_M_t_1 =  nn.MM()({addrCalcTransp, initialMem})
    local resAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
    local AAT_M_t_1 = nn.MM()({resAddrCalc, AT_M_t_1})

    --memory update
    local memEraser = nn.CSubTable()({initialMem, AAT_M_t_1})
    local finMem = nn.CAddTable()({memEraser, adder})
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    -- Probability calculator
    ----------------------------------------------------------------------------
    local addrValCalc = nn.JoinTable(1)({addrCalc, valueCalc})
    local allInOne = nn.JoinTable(1)({addrValCalc, reshapedMem})
    local h1 = nn.Linear(vectorSize + memSize + memSize * vectorSize, 10)
        (allInOne)

    local p = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
        nn.Sigmoid()(h1)))))
    ----------------------------------------------------------------------------

    return nn.gModule({initialMem, input}, {finMem, p})

end


--------------------------------------------------------------------------------
-- Creates model based only on memory
--------------------------------------------------------------------------------
function Model.__createNoInput(opt)
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)

    ----------------------------------------------------------------------------
    --  Initial Memory
    ----------------------------------------------------------------------------
    local initialMem = nn.Identity()()
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    --  Address Encoder
    ----------------------------------------------------------------------------
    local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
    local enc = nn.GRU(memSize * vectorSize, memSize)(reshapedMem)
    local address = nn.SoftMax()(enc)
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    -- Value Extractor
    ----------------------------------------------------------------------------
    --TODO for current problems may not need to send value here
    local addressTransp = nn.Reshape(1, memSize)(address)
    --return nn.gModule({initialMem}, {addressTransp})

    local value = nn.MM()({addressTransp, initialMem})
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    ---- Next address calculator
    ----------------------------------------------------------------------------
    local reshapedValue = nn.Squeeze(1)(value)
    local valAddr = nn.JoinTable(1)({address, reshapedValue})
    local addrCalc =
        nn.GRU(memSize + vectorSize, memSize)(valAddr)
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    ---- Next value calculator
    ----------------------------------------------------------------------------
    local valueCalc =
        nn.GRU(memSize + vectorSize, vectorSize)(valAddr)
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    ---- Memory Calculator
    ----------------------------------------------------------------------------

    ----adder
    local resizeValueCalc = nn.Reshape(1, vectorSize)(valueCalc)
    local resizeAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
    local adder = nn.MM()({resizeAddrCalc, resizeValueCalc})


    ---- eraser
    local addrCalcTransp = nn.Reshape(1, memSize)(addrCalc)
    local AT_M_t_1 =  nn.MM()({addrCalcTransp, initialMem})
    local resAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
    local AAT_M_t_1 = nn.MM()({resAddrCalc, AT_M_t_1})

    ----memory update
    local memEraser = nn.CSubTable()({initialMem, AAT_M_t_1})
    local finMem = nn.Sigmoid()(nn.CAddTable()({memEraser, adder}))
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    ---- Probability calculator
    ----------------------------------------------------------------------------
    local addrValCalc = nn.JoinTable(1)({addrCalc, valueCalc})
    local allInOne = nn.JoinTable(1)({addrValCalc, reshapedMem})
    local h1 = nn.Linear(vectorSize + memSize + memSize * vectorSize, 10)
        (allInOne)

    local p = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
        nn.Sigmoid()(h1)))))
    ----------------------------------------------------------------------------

    return nn.gModule({initialMem}, {finMem, p})

end

--------------------------------------------------------------------------------
-- Save model to file
-- Specify overWrite = true if you wish to overwrite an existent file
--------------------------------------------------------------------------------
function Model.saveModel(model, fileName, overWrite)
    --TODO remove hardcoding
    if fileName == nil then
        fileName = "autosave.model"
    end
    if (path.exists(fileName) and overWrite == false) then
        print("file "..fileName.." already exists, overWrite option not specified. aborting.")
        return false
    end
    torch.save(fileName, model)
    print("Saved model!")
    return true
end

--------------------------------------------------------------------------------
-- Load a model from a file
--------------------------------------------------------------------------------
function Model.loadModel(fileName)
    if not path.exists(fileName) then
        print("file "..fileName.." does not exist. Create it first before loading something from it")
        return nil
    end
    model = torch.load(fileName)
    return model
end


return Model
