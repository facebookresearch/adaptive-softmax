-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tnt = require 'torchnet.env'

local RNNDataset, _ = torch.class("tnt.RNNDataset", "tnt.Dataset", tnt)

function RNNDataset:__init(data, bsz, bptt)
   local ntoken = data:nElement()
   local nbatch = math.ceil(ntoken / (bsz * bptt))
   local tsize  = nbatch * bptt * bsz
   local buffer = torch.LongTensor(tsize):fill(1)
   buffer:narrow(1, tsize - ntoken + 1, ntoken):copy(data)
   buffer = buffer:view(bsz, nbatch * bptt):t()
   self.nbatch = nbatch
   self.bsz    = bsz
   self.bptt   = bptt
   self.data   = torch.LongTensor(nbatch * bptt + 1, bsz):fill(1)
   self.data:narrow(1, 2, nbatch * bptt):copy(buffer)
end

function RNNDataset:size()
   return self.nbatch
end

function RNNDataset:get(i)
   local pos    = 1 + self.bptt * (i - 1)
   local input  = torch.LongTensor(self.bptt, self.bsz)
   local target = torch.LongTensor(self.bptt, self.bsz)
   input:copy(self.data:narrow(1, pos, self.bptt))
   target:copy(self.data:narrow(1, pos+1, self.bptt))
   return {input = input, target = target}
end
