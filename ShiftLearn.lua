require "torch"
require "nn"
require "nngraph"
nngraph.setDebug(true)
nngraph.annotateNodes()
local class = require("class")


-- static class
local ShiftLearn = class("ShiftLearn")

function ShiftLearn.create(vecSize)

   -----------------------------------------------------------------------------
   -- Input def
   -----------------------------------------------------------------------------
   local sh = nn.Identity()()                                   -- shift command
   local x = nn.Identity()()                                    -- input address

   -----------------------------------------------------------------------------
   -- Internal shift matrix
   -----------------------------------------------------------------------------
   -- false refers to the fact that we DO NOT want bias
   local learner2D = nn.Linear(vecSize, vecSize * vecSize, false)(sh)

   -----------------------------------------------------------------------------
   -- Shifted Tensor
   -----------------------------------------------------------------------------

   local fin = nn.MM()({
         nn.Reshape(1, vecSize)(x),
         nn.Reshape(vecSize, vecSize)(learner2D)
   })
   local res_fin = nn.Squeeze()(fin)    -- SHARPEN!?

   return nn.gModule({sh, x}, {res_fin})

end

function ShiftLearn.createDummy(vecSize, shiftNo)

   -----------------------------------------------------------------------------
   -- Input def
   -----------------------------------------------------------------------------
   local sh = nn.Identity()()                                   -- shift command
   local addr = nn.Identity()()                                 -- input address

   -----------------------------------------------------------------------------
   -- Internal shift matrix
   -----------------------------------------------------------------------------
   local I = torch.eye(vecSize)
   local J
   shiftNo = shiftNo % vecSize
   if shiftNo > 0 then
      J = torch.cat(
         I[{{},{vecSize-shiftNo+1, vecSize}}],
         I[{{},{1, vecSize-shiftNo}}]
      )
   else
      J = I
   end
   print(J)
   local learner2D = nn.Constant(J)(sh)

   -----------------------------------------------------------------------------
   -- Shifted Tensor
   -----------------------------------------------------------------------------

   local fin = nn.MM()({
         nn.Reshape(1, vecSize)(addr),
         nn.Reshape(vecSize, vecSize)(learner2D)
   })
   local res_fin = nn.Squeeze()(fin)    --Sharpen

   return nn.gModule({sh, addr}, {res_fin})

end



function ShiftLearn.createWrapper(vecSize, shiftNo)
   -- shift address input
   print(shiftNo)
   -----------------------------------------------------------------------------
   -- Internal shift generator
   -----------------------------------------------------------------------------
   local shifter
   if shiftNo then
      shifter = ShiftLearn.createDummy(vecSize, shiftNo)
   else
      shifter = ShiftLearn.create(vecSize)
   end

   -----------------------------------------------------------------------------
   -- Currently shift amount is constant
   -----------------------------------------------------------------------------
   local shift_address = nn.Identity()()

   local dep_vec = torch.zeros(vecSize)
   dep_vec[1] = 1

   local dep_constant = nn.Constant(dep_vec)(shift_address)

   local shift_wrapper = shifter({dep_constant, shift_address})

   return nn.gModule({shift_address}, {shift_wrapper})

end

return ShiftLearn
