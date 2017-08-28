-- Copyright 2004-present Facebook. All rights reserved.
-- Author: Edouard Grave <egrave@fb.com>

local tablex = require 'pl.tablex'
local stringx = require 'pl.stringx'

local word2vec = {}

function word2vec.load(filename)
   local data = {}
   local file = assert(io.open(filename), 'Unable to open file '..filename)
   data.__size__ = tablex.map(tonumber, stringx.split(file:read()))
   for line in file:lines() do
      local tline = stringx.split(line)
      local word  = tline[1]
      local vec   = tablex.map(tonumber, tablex.sub(tline, 2, #tline))
      data[word] = torch.Tensor(vec)
   end
   file:close()
   return data
end

return word2vec
