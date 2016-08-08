require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'
require 'optim'
require 'cutorch'
require 'image'
require 'gnuplot'


torch.manualSeed(0)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Plot values from file, each val on separate line')
cmd:text()
cmd:text('Options')
cmd:option('-plotFile', "defaultPlotFiles", 'filename of the dataset to plot')


opt = cmd:parse(arg or {})
file = io.open(opt.plotFile, "rb")
local current = file:read("*all")
file:close()

local tabel_vals = {}
local tabel_ix = {}

for token in string.gmatch(current, "[^%s]+") do
   tabel_ix[#tabel_ix + 1] = #tabel_ix + 1
   tabel_vals[#tabel_vals + 1] = tonumber(token) + (torch.random() % 100)
end

local tensor = torch.Tensor(tabel_vals)
local ix = torch.Tensor(tabel_ix)

gnuplot.figure(1)
gnuplot.plot(ix, tensor)
gnuplot.plotflush()
