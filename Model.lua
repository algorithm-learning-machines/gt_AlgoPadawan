--------------------------------------------------------------------------------
-- File containing model definition
--------------------------------------------------------------------------------
require "rnn"
require "nngraph"
--require "cutorch"
--require "cunn"
local Model = {}


--------------------------------------------------------------------------------
-- Creates model
--------------------------------------------------------------------------------
function Model.create(opt)
    local vectorSize = tonumber(opt.vectorSize)
    local memSize = tonumber(opt.memorySize)
    print(memSize)
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
    --TODO for current problems may not need to send value here
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
    local p = nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(nn.Sigmoid()(h1))))
    ----------------------------------------------------------------------------

    return nn.gModule({initialMem, input}, {finMem, p})

end

return Model
