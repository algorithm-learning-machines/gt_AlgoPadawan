--------------------------------------------------------------------------------
-- File containing model definition
--------------------------------------------------------------------------------
require "rnn" require "nngraph"
local Model = {}


--------------------------------------------------------------------------------
-- Creates model, may use external input besides
--------------------------------------------------------------------------------
function Model.create(opt)
   if opt.noInput == true then
      if opt.noProb then
         return Model.__createNoInputOrProb(opt)
      else
         if opt.NRAMProb then
            print("giig")
            return Model.__createNoInputNRAMProb(opt)
         else
            return Model.__createNoInput(opt)
         end
      end
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
   local RNN_steps = 5 --TODO add command line param

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
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_steps)(reshapedMem)
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
   nn.GRU(inputSize + memSize + vectorSize, memSize, RNN_steps)
   (inputValAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   -- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(inputSize + memSize + vectorSize, vectorSize, RNN_steps)
   (inputValAddr)
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
   local RNN_size = 5 -- TODO add command line param
   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_size)(reshapedMem)
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
   ---- Next address calculator
   ----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)
   local valAddr = nn.JoinTable(1)({address, reshapedValue})
   local addrCalc =
   nn.GRU(memSize + vectorSize, memSize, RNN_size)(valAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   ---- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(memSize + vectorSize, vectorSize, RNN_size)(valAddr)
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
-- Creates model based only on memory; no probability
--------------------------------------------------------------------------------
function Model.__createNoInputOrProb(opt)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local RNN_size = 5 -- TODO add command line param
   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_size)(reshapedMem)
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
   ---- Next address calculator
   ----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)
   local valAddr = nn.JoinTable(1)({address, reshapedValue})
   local addrCalc =
   nn.GRU(memSize + vectorSize, memSize, RNN_size)(valAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   ---- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(memSize + vectorSize, vectorSize, RNN_size)(valAddr)
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

   return nn.gModule({initialMem}, {finMem})

end


-- TODO add input as well
-- TODO generalize model, for all cases, not that hard

--------------------------------------------------------------------------------
-- Creates model based only on memory, simulates NRAM probability
--------------------------------------------------------------------------------
function Model.__createNoInputNRAMProb(opt)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local RNN_size = 5 -- TODO add command line param

   ----------------------------------------------------------------------------
   -- Previous timestep probability factor, function of p_t-1 and f_t-1
   ----------------------------------------------------------------------------
   local prevDelta = nn.Identity()()
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_size)(reshapedMem)
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
   ---- Next address calculator
   ----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)
   local valAddr = nn.JoinTable(1)({address, reshapedValue})
   local addrCalc =
   nn.GRU(memSize + vectorSize, memSize, RNN_size)(valAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   ---- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(memSize + vectorSize, vectorSize, RNN_size)(valAddr)
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


   ----------------------------------------------------------------------------
   -- P and F are both integrated in model for derivative simplicity
   ----------------------------------------------------------------------------
   local f = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
   nn.Sigmoid()(h1)))))

   local p = nn.MM()({nn.Reshape(1,1)(f), nn.Reshape(1,1)(prevDelta)})
   ----------------------------------------------------------------------------
   --
   return nn.gModule({initialMem, prevDelta}, {finMem, f, p})

end

--------------------------------------------------------------------------------
-- Refactored model creation
--------------------------------------------------------------------------------
function Model.createUniversal(opt)
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
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Input
   ----------------------------------------------------------------------------
   local input = nil

   if not opt.noInput then
      input = nn.Identity()()
   end
   ----------------------------------------------------------------------------
   

   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_steps)(reshapedMem)
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
   local inputValAddr = nil
   if not opt.noInput then
      local inputVal = nn.JoinTable(1)({input, address})
      inputValAddr = nn.JoinTable(1)({inputVal, reshapedValue})
   else
      inputValAddr = nn.JoinTable(1)({address, reshapedValue})
   end
   local addrCalc =
   nn.GRU(inputSize + memSize + vectorSize, memSize, RNN_steps)
   (inputValAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   -- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(inputSize + memSize + vectorSize, vectorSize, RNN_steps)
   (inputValAddr)
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

   if opt.noProb then
      if opt.noInput then
         return nn.gModule({initialMem}, {finMem})
      end
      return nn.gModule({initialMem, input}, {finMem})
   end

   ----------------------------------------------------------------------------
   -- Probability calculator
   ----------------------------------------------------------------------------
   local addrValCalc = nn.JoinTable(1)({addrCalc, valueCalc})
   local allInOne = nn.JoinTable(1)({addrValCalc, reshapedMem})
   local h1 = nn.Linear(vectorSize + memSize + memSize * vectorSize, 10)
   (allInOne)

   local p = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
   nn.Sigmoid()(h1)))))

   if opt.NRAMProb then
      local prevDelta = nn.Identity()()
      local pNRAM = nn.MM()({nn.Reshape(1,1)(p), nn.Reshape(1,1)(prevDelta)})
      if opt.noInput then
         return nn.gModule({initialMem, prevDelta}, {finMem, p, pNRAM})
      end
      return nn.gModule({initialMem, input, prevDelta}, {finMem, p, pNRAM})

   end
   ----------------------------------------------------------------------------

   return nn.gModule({initialMem, input}, {finMem, p})

end


--------------------------------------------------------------------------------
-- Creates model based only on memory
--------------------------------------------------------------------------------
function Model.__createNoInput(opt)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local RNN_size = 5 -- TODO add command line param
   ----------------------------------------------------------------------------
   --  Initial Memory
   ----------------------------------------------------------------------------
   local initialMem = nn.Identity()()
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   local enc = nn.GRU(memSize * vectorSize, memSize, RNN_size)(reshapedMem)
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
   ---- Next address calculator
   ----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)
   local valAddr = nn.JoinTable(1)({address, reshapedValue})
   local addrCalc =
   nn.GRU(memSize + vectorSize, memSize, RNN_size)(valAddr)
   ----------------------------------------------------------------------------


   ----------------------------------------------------------------------------
   ---- Next value calculator
   ----------------------------------------------------------------------------
   local valueCalc =
   nn.GRU(memSize + vectorSize, vectorSize, RNN_size)(valAddr)
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
