require "torch"
require "nn"

m1 = nn.Linear(5,2)
m2 = nn.Linear(5,2)
m3 = nn.Linear(5,2)


W = torch.linspace(0, 1, 10):reshape(2, 5)
b = torch.zeros(2)

m1:share({["weight"]=W, ["bias"]=b}, "weight", "bias")
m2:share({["weight"]=W, ["bias"]=b}, "weight", "bias")
m3:share({["weight"]=W, ["bias"]=b}, "weight", "bias")

y1 = m1:forward(torch.ones(5))
y2 = m2:forward(torch.ones(5))
y3 = m3:forward(torch.ones(5))

m1:zeroGradParameters()
m2:zeroGradParameters()
m3:zeroGradParameters()

local _mse = nn.MSECriterion()
local _nll = nn.ClassNLLCriterion()

local mse = nn.MSECriterion()
local nll = nn.ClassNLLCriterion()

local dis = 0.7
local par = nn.ParallelCriterion():add(_mse):add(_nll)

d1 = mse:backward(y1, torch.ones(2))
d2 = nll:backward(y2, torch.ones(2))

d3 = par:backward({y1, y1}, {torch.ones(2), torch.ones(2)})
print(d3[1])
print(d3[2])
print(d1)
print(d2)
--print()
--print(y)
