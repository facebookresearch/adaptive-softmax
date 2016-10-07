-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local tds      = require 'tds'
local stringx  = require 'pl.stringx'
local tablex   = require 'pl.tablex'

require 'data.RNNDataset'

local data = {}

data.addword = function(dic, word, count)
   local count = count or 1
   if dic.word2idx[word] == nil then
      dic.idx2word:insert(word)
      dic.word2idx[word] = #dic.idx2word
      dic.idx2freq[#dic.idx2word] = count
   else
      local idx = dic.word2idx[word]
      dic.idx2freq[idx] = dic.idx2freq[idx] + count
   end
   return dic
end

data.getidx = function(dic, word)
   return dic.word2idx[word] or dic.word2idx["<unk>"]
end

data.initdictionary = function()
   local dic = {
      idx2word = tds.Vec(),
      word2idx = tds.Hash(),
      idx2freq = {},
   }
   data.addword(dic, "</s>")
   data.addword(dic, "<unk>")
   return dic
end

data.savedictionary = function(dic, filename)
   print(filename)
   local fout = io.open(filename, 'w')
   for i = 1, #dic.idx2word do
      fout:write(i ..' '.. dic.idx2word[i] ..' '.. dic.idx2freq[i] ..'\n')
   end
   fout:close()
end

data.loaddictionary = function(filename)
   local dic = data.initdictionary()
   for line in io.lines(filename) do
      local tokens = stringx.split(line)
      local idx = tonumber(tokens[1])
      local freq = tonumber(tokens[3])
      local word = tokens[2]
      dic.idx2word[idx] = word
      dic.word2idx[word] = idx
      dic.idx2freq[idx] = freq
   end
   dic.idx2freq = torch.Tensor(dic.idx2freq)
   return dic
end

data.makedictionary = function(filename)
   local dic = data.initdictionary()
   local lines = 0

   for line in io.lines(filename) do
      local words = stringx.split(line)
      tablex.map(function(w) data.addword(dic, w) end, words)
      lines = lines + 1

      if lines % 10000 == 0 then
         collectgarbage()
      end
   end
   dic.idx2freq[dic.word2idx["</s>"]] = lines
   dic.idx2freq[dic.word2idx["<unk>"]] = dic.idx2freq[2] ~= 0
      and dic.idx2freq[2] or 1 -- nonzero hack

   dic.idx2freq = torch.Tensor(dic.idx2freq)
   print(string.format("| Dictionary size %d", #dic.idx2word))
   return dic
end

data.sortthresholddictionary = function(dic, threshold)
   local freq, idxs = dic.idx2freq:sort(true)
   local newdic = data.initdictionary()

   for i = 1, idxs:size(1) do
      if freq[i] <= threshold then
         break
      end
      data.addword(newdic, dic.idx2word[idxs[i]], freq[i])
   end
   newdic.idx2freq = torch.Tensor(newdic.idx2freq)

   collectgarbage()
   collectgarbage()
   print(string.format("| Dictionary size %d", #newdic.idx2word))

   return newdic
end

local function add_data_to_tensor(tensor, buffer)
   if tensor then
      if #buffer > 0 then
         return torch.cat(tensor, torch.LongTensor(buffer))
      else
         return tensor
      end
   else
      return torch.LongTensor(buffer)
   end
end

data.loadfile = function(filename, dic)
   local buffer = {}
   local tensor = nil

   for line in io.lines(filename) do
      local words = stringx.split(line)
      table.insert(words, "</s>")
      local idx = tablex.map(function(w) return data.getidx(dic, w) end, words)
      tablex.insertvalues(buffer, idx)

      if #buffer > 5000000 then
         tensor = add_data_to_tensor(tensor, buffer)
         buffer = {}
         collectgarbage()
      end
   end
   tensor = add_data_to_tensor(tensor, buffer)

   print(string.format("| Load file %s: %d tokens",
                       filename, tensor:size(1)))
   collectgarbage()

   return tensor
end

return data
