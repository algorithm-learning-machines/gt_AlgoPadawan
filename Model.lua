--------------------------------------------------------------------------------
-- File containing model definition --------------------------------------------
--------------------------------------------------------------------------------
require "nn"
require "rnn"
require "nngraph"
require "ShiftLearn"
local Model = {}


function Model.createDebug(opt, addressReader, addressWriter, valueWriter)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local inputSize = 0
   if not opt.noInput then
      inputSize = tonumber(opt.inputSize)
   end

   local RNN_steps = 5 --TODO add command line param

   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   --local initialMem = nil
   --initialMem = nn.Identity()()
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Input
   ----------------------------------------------------------------------------
   local input = nil

   if not opt.noInput then
      input = nn.Identity()()
   end
   ----------------------------------------------------------------------------

   -----------------------------------------------------------------------------
   -- Previous write address
   -----------------------------------------------------------------------------
   local prevWriteAddress = nn.Identity()()

   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   -- local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)

   params = {memSize * vectorSize, memSize, RNN_steps}
   local dummyInput = nn.Identity()()
   if addressReader then
      AR = addressReader
      params = {memSize}
      linkedNode = {dummyInput, prevWriteAddress}
   end

   local enc = addressReader({dummyInput, prevWriteAddress})
   local address = nn.SoftMax()(enc)
   return nn.gModule({dummyInput, prevWriteAddress}, {address})

end


--------------------------------------------------------------------------------
-- !! Order of modules at end
-- [initialMem, input, prevAddrWrite] -> [finMem, addrCalc, p, pNRAM]
--------------------------------------------------------------------------------
function Model.create(opt, addressReader, addressWriter, valueWriter)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local inputSize = 0
   if not opt.noInput then
      inputSize = tonumber(opt.inputSize)
   end
   local dummyInput = nn.Identity()()
   local RNN_steps = 5 --TODO add command line param

   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Input
   ----------------------------------------------------------------------------
   local input

   if not opt.noInput then
      input = nn.Identity()()
   end
   ----------------------------------------------------------------------------

   -----------------------------------------------------------------------------
   -- Previous write address
   -----------------------------------------------------------------------------
   local prevWriteAddress = nn.Identity()()

   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nil
   if not addressReader then
      reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   end

   local AR = nn.GRU
   params = {memSize * vectorSize, memSize, RNN_steps}
   linkedNode = reshapedMem

   if addressReader then
      AR = addressReader
      params = {memSize}
      linkedNode = {prevWriteAddress}
   end

   local enc = AR(unpack(params))(linkedNode)
   local address = nn.SoftMax()(enc)
   -----------------------------------------------------------------------------


   -----------------------------------------------------------------------------
   -- Value Extractor
   -----------------------------------------------------------------------------

   local addressTransp = nn.Reshape(1, memSize)(address)
   local value = nn.MM()({addressTransp, initialMem})     -- extract memory line
   -----------------------------------------------------------------------------

   -----------------------------------------------------------------------------
   -- Next address calculator
   -----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)             -- a line should be 1D
   local inputValAddr
   local inputVal
   local inputAddr
   if opt.separateValAddr then                -- separate value and adress paths
      if opt.noInput then
         inputVal = reshapedValue
         inputAddr = address
      else
         inputVal = nn.JoinTable(1)({input, reshapedValue})
         inputAddr = nn.JoinTable(1)({input, address})
      end
   else                                          -- cross value and adress paths
      if not pt.noInput then                            -- TODO: invert if cases
         local auxJoin = nn.JoinTable(1)({input, address})
         inputVal = nn.JoinTable(1)({auxJoin, reshapedValue})
         inputAddr = inputVal
      else
         inputVal = nn.JoinTable(1)({address, reshapedValue})
         inputAddr = inputVal
      end
   end

   local AW = nn.GRU

   params = {inputSize + memSize, memSize, RNN_steps}
   if not opt.separateValAddr then
      params = {inputSize + memSize + vectorSize, memSize, RNN_steps}
   end

   linkedNode = inputAddr
   if addressWriter then
      AW = addressWriter
      params = {memSize}
   end

   local addrCalc = AW(unpack(params))(linkedNode)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   -- Next value calculator
   ----------------------------------------------------------------------------

   -- TODO add custom value writer
   local VW = nn.GRU
   params = {inputSize + vectorSize, vectorSize, RNN_steps}

   if not opt.separateValAddr then
      params = {inputSize + memSize + vectorSize, vectorSize, RNN_steps}
   end

   linkedNode = inputVal
   if valueWriter then
      VW = valueWriter
      params = {}
   end

   local valueCalc = VW(unpack(params))(linkedNode)
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Memory Calculator
   ----------------------------------------------------------------------------

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
   ----------------------------------------------------------------------------

   in_dict = {}
   out_dict = {}
   in_dict[#in_dict + 1] = initialMem
   out_dict[#out_dict + 1] = finMem
   if not opt.noInput then -- add input to initial dict
      in_dict[#in_dict + 1] = input
   end
   if addressReader then -- add back address to input
      in_dict[#in_dict + 1] = prevWriteAddress
      out_dict[#out_dict + 1] = addrCalc
   end

   if opt.noProb then
      return nn.gModule(in_dict, out_dict)
   end

   -----------------------------------------------------------------------------
   -- Probability calculator
   -----------------------------------------------------------------------------
   local addrValCalc = nn.JoinTable(1)({addrCalc, valueCalc})
   local allInOne = nn.JoinTable(1)({addrValCalc, reshapedMem})

   --TODO maybe this could be generalized as well
   local h1 = nn.Linear(vectorSize + memSize + memSize * vectorSize, 10)
   (allInOne) -- hidden layer

   local p = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
   nn.Sigmoid()(h1)))))

   out_dict[#out_dict + 1] = p
   -----------------------------------------------------------------------------
   -- NRAM probability calculator
   -----------------------------------------------------------------------------

   if opt.NRAMProb then
      local prevDelta = nn.Identity()()
      local pNRAM = nn.MM()({nn.Reshape(1,1)(p), nn.Reshape(1,1)(prevDelta)})
      out_dict[#out_dict + 1] = pNRAM
   end
   ----------------------------------------------------------------------------

   return nn.gModule(in_dict, out_dict)
end

--------------------------------------------------------------------------------
-- Save model to file
-- Specify overWrite = true if you wish to overwrite an existent file
--------------------------------------------------------------------------------
function Model.saveModel(model, fileName, overWrite)
   --TODO remove hardcoding
   --if fileName == nil then
      --fileName = "autosave.model"
   --end
   --if (path.exists(fileName) and overWrite == false) then
      --print("file "..fileName.." already exists, overWrite option not specified. aborting.")
      --return false
   --end
   --torch.save(fileName, model)
   --print("Saved model!")
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
