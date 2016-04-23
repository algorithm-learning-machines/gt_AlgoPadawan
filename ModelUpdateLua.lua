-- Confirm that memory update module works correctly
require "torch"
require "nn"
require "nngraph"


-- initial mem -> M * N
-- valueCalc -> N
-- addrCalc -> M

local vectorSize = 5 
local memSize = 10

local valueCalc = nn.Identity()()
local addrCalc = nn.Identity()()
local initialMem = nn.Identity()()

-- adder
local resizeValueCalc = nn.Reshape(1, vectorSize)(valueCalc)
local resizeAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
local adder = nn.MM()({resizeAddrCalc, resizeValueCalc})

-- eraser
local addrCalcTransp = nn.Reshape(1, memSize)(addrCalc)
local AT_M_t_1 =  nn.MM()({addrCalcTransp, initialMem})
local resAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
local AAT_M_t_1 = nn.MM()({resAddrCalc, AT_M_t_1})

-- memory update
local memEraser = nn.CSubTable()({initialMem, AAT_M_t_1})
local finMem = nn.CAddTable()({memEraser, adder})


local mod = nn.gModule({addrCalc, valueCalc, initialMem}, {finMem}) 

-- test
local mem = torch.zeros(memSize, vectorSize)
mem[1] = torch.Tensor{1,0,1,0,1}
mem[2] = torch.ones(vectorSize)
local val = torch.zeros(vectorSize)
val[3] = 1
local address = torch.zeros(memSize)
address[2] = 1
print("Initial memory:")
print(mem)
print("Value that shall be written:")
print(val)
print("Address at which value shall be written:")
print(address)
print("Memory after update:")
print(mod:forward({address, val, mem}))
