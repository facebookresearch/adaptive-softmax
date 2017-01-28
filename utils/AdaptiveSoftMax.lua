-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tablex = require 'pl.tablex'

local AdaptiveSoftMax, Module = torch.class('nn.AdaptiveSoftMax', 'nn.Module')

function AdaptiveSoftMax:__init(isz, cutoff)
   Module.__init(self)

   self.isz    = isz
   self.cutoff = cutoff
   self.hsz    = cutoff[1] + #cutoff - 1

   self.head = nn.Linear(isz, self.hsz, false)
   self.tail = {}
   for i = 1, #cutoff - 1 do
      local seq = nn.Sequential()
      seq:add( nn.Linear(isz, isz / math.pow(4, i), false) )
      seq:add( nn.Linear(isz / math.pow(4, i), cutoff[i+1] - cutoff[i], false) )
      self.tail[i] = seq
   end
   return self
end

function AdaptiveSoftMax:reset(stdv)
   local stdv = stdv or 1.0 / math.sqrt(self.isz)
   self.head.weight:uniform(-stdv, stdv)
   for i = 1, #self.tail do
      self.tail[i]:get(1).weight:uniform(-stdv, stdv)
      self.tail[i]:get(2).weight:uniform(-stdv, stdv)
   end
   return self
end

function AdaptiveSoftMax:setTarget(target)
   local cutoff = self.cutoff
   self.idx = {}
   for i = 1, #cutoff - 1 do
      local m = target:ge(cutoff[i]+1):cmul(target:le(cutoff[i+1]))
      if m:any() then
         -- the nonzero function is not implemented for CudaTensor :(
         table.insert(self.idx, m:float():nonzero():squeeze(2))
      else
         table.insert(self.idx, {})
      end
   end
end

function AdaptiveSoftMax:updateOutput(input)
   self.output = {}

   self.head:updateOutput(input)
   table.insert(self.output, self.head.output)

   for i = 1, #self.idx do
      if torch.isTensor(self.idx[i]) then
         self.tail[i]:updateOutput(input:index(1, self.idx[i]))
         table.insert(self.output, self.tail[i].output)
      else
         table.insert(self.output, {})
      end
   end

   return self.output
end

function AdaptiveSoftMax:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)

   self.head:updateGradInput(input, gradOutput[1])
   self.gradInput:copy(self.head.gradInput)

   for i = 1, #self.idx do
      if torch.isTensor(self.idx[i]) then
         self.tail[i]:updateGradInput(input:index(1, self.idx[i]), gradOutput[i+1])
         self.gradInput:indexAdd(1, self.idx[i], self.tail[i].gradInput)
      end
   end

   return self.gradInput
end

function AdaptiveSoftMax:accGradParameters(input, gradOutput)
   self.head:accGradParameters(input, gradOutput[1])

   for i = 1, #self.idx do
      if torch.isTensor(self.idx[i]) then
         self.tail[i]:accGradParameters(input:index(1, self.idx[i]), gradOutput[i+1])
      end
   end
end

function AdaptiveSoftMax:accUpdateParameters(input, gradOutput)
   error('Function AdaptiveSoftMax:accUpdateParameters not implemented')
end

function AdaptiveSoftMax:parameters()
   local params = {self.head.weight}
   local grads  = {self.head.gradWeight}
   for i = 1, #self.tail do
      local tail_params, tail_grads = self.tail[i]:parameters()
      tablex.insertvalues(params, tail_params)
      tablex.insertvalues(grads, tail_grads)
   end
   return params, grads
end

function AdaptiveSoftMax:zeroGradParameters()
   self.head:zeroGradParameters()
   for i = 1, #self.tail do
      self.tail[i]:zeroGradParameters()
   end
end

function AdaptiveSoftMax:updateParameters(learningRate)
   self.head:updateParameters(learningRate)
   for i = 1, #self.tail do
      self.tail[i]:updateParameters(learningRate)
   end
end

function AdaptiveSoftMax:getLogProb(input)
   local lsm   = nn.LogSoftMax():cuda()

   self.head:updateOutput(input)

   local bsz   = self.head.output:size(1)
   local proba = torch.zeros(bsz, self.cutoff[#self.cutoff]):cuda()

   lsm:updateOutput(self.head.output)
   proba:narrow(2, 1, self.hsz):add(lsm.output:narrow(2, 1, self.hsz))

   for i = 1, #self.tail do
      local pos = self.cutoff[i] + 1
      local tsz = self.cutoff[i+1] - self.cutoff[i]
      local buffer = lsm.output:narrow(2, self.cutoff[1] + i, 1)
      buffer = buffer:expand(bsz, tsz)
      proba:narrow(2, pos, tsz):copy(buffer)
   end

   for i = 1, #self.tail do
      local pos = self.cutoff[i] + 1
      local tsz = self.cutoff[i+1] - self.cutoff[i]
      self.tail[i]:updateOutput(input)
      lsm:updateOutput(self.tail[i].output)
      proba:narrow(2, pos, tsz):add(lsm.output)
   end

   return proba
end
