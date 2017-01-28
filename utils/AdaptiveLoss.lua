-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local AdaptiveLoss, Criterion = torch.class('nn.AdaptiveLoss', 'nn.Criterion')

function AdaptiveLoss:__init(cutoff)
   Criterion.__init(self)
   self.cutoff = cutoff
   self.criterions = {}
   for i = 1, #cutoff do
      table.insert(self.criterions, nn.CrossEntropyCriterion())
      self.criterions[i].nll.sizeAverage = false
   end
end

function AdaptiveLoss:remapTarget(target)
   local new_target = {target:clone()}
   local cutoff = self.cutoff
   for i = 1, #cutoff - 1 do
      local m = target:ge(cutoff[i] + 1):cmul(target:le(cutoff[i+1]))
      new_target[1][m] = cutoff[1] + i
      if m:any() then
         table.insert(new_target, target[m]:add(-cutoff[i]))
      else
         table.insert(new_target, false)
      end
   end
   return new_target
end

function AdaptiveLoss:updateOutput(input, target)
   local bsz = input[1]:size(1)
   local target = self:remapTarget(target)

   self.output = 0.0
   self.gradInput = {}

   for i = 1, #input do
      if input[i] then
         assert(target[i]:min() > 0 and target[i]:max() <= input[i]:size(2))
         local criterion = self.criterions[i]
         self.output = self.output + criterion:updateOutput(input[i], target[i])
         self.gradInput[i] = criterion:updateGradInput(input[i], target[i])
         self.gradInput[i]:mul(1.0 / bsz)
      end
   end
   self.output = self.output / bsz
   return self.output
end

function AdaptiveLoss:updateGradInput(input, target)
   return self.gradInput
end

function AdaptiveLoss:cuda()
   for i = 1, #self.criterions do
      self.criterions[i]:cuda()
   end
   return self
end
