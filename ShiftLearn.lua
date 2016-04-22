require "torch"
require "nn"
require "nngraph"
nngraph.setDebug(true)
nngraph.annotateNodes()
local class = require("class")


-- static class
local ShiftGenerator = class("ShiftLearn")

function ShiftGenerator.create(vecSize)

   -----------------------------------------------------------------------------
   -- Input def
   -----------------------------------------------------------------------------
   local sh = nn.Identity()()
   local x = nn.Identity()()
   --local x_sh = nn.JoinTable(1)({sh, x})

   -----------------------------------------------------------------------------
   -- Internal shift matrix
   -----------------------------------------------------------------------------
   local learner2D = 
      nn.Linear(vecSize, vecSize * vecSize)(sh)

   -----------------------------------------------------------------------------
   -- Shifted Tensor
   -----------------------------------------------------------------------------

   local fin = nn.MM()({nn.Reshape(1, vecSize)(x),
      nn.Reshape(vecSize, vecSize)(learner2D)})
   local res_fin = nn.SoftMax()(nn.Squeeze()(fin))

   return nn.gModule({sh, x}, {res_fin})

end

function ShiftGenerator.createWrapper(vecSize)
   -- shift address input

   -----------------------------------------------------------------------------
   -- Internal shift generator
   -----------------------------------------------------------------------------
   local shifter = ShiftGenerator.create(vecSize)

   -----------------------------------------------------------------------------
   -- Currently shift amount is constant
   -----------------------------------------------------------------------------
   local shift_address = nn.Identity()()

   local dep_vec = torch.zeros(vecSize)
   dep_vec[1] = 1

   local dep_constant = nn.Constant(dep_vec)(shift_address)

   local shift_wrapper = shifter({dep_constant, shift_address})

   --concat table like in the example
   return nn.gModule({shift_address}, {shift_wrapper})

end

return ShiftGenerator 
