-- Copyright (c) 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require 'math'
require 'cutorch'
require 'nn'
require 'cunn'
require 'rnnlib'

local tablex  = require 'pl.tablex'
local stringx = require 'pl.stringx'
local tnt     = require 'torchnet'
local optim   = require 'optim'
local data    = require 'data'
local utils   = require 'utils'

torch.setheaptracking(true)

local cmd = torch.CmdLine('-', '-')
cmd:option('-seed', 1111, 'Seed for the random generator')

cmd:option('-isz',     128, 'Dimension of input word vectors')
cmd:option('-nhid',    128, 'Number of hidden variables per layer')
cmd:option('-nlayer',  1,   'Number of layers')
cmd:option('-dropout', 0.0, 'Dropout probability')

cmd:option('-lr',            0.1,  'Learning rate')
cmd:option('-epsilon',       1e-5, 'Epsilon for Adagrad')
cmd:option('-initrange',     0.1,  'Init range')
cmd:option('-maxepoch',      10,   'Number of epochs')
cmd:option('-bptt',          20,   'Number of backprop through time steps')
cmd:option('-clip',          0.25, 'Threshold for gradient clipping')
cmd:option('-batchsize',     16,   'Batch size')
cmd:option('-testbatchsize', 16,   'Batch size for test')

cmd:option('-data',      '', 'Path to the dataset directory')
cmd:option('-outdir',    '', 'Path to the output directory')
cmd:option('-threshold', 0,  'Threshold for <unk> words')

cmd:option('-cutoff', '',   'Cutoff for AdaptiveSoftMax')

cmd:option('-usecudnn', false, '')

local config = cmd:parse(arg)

torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)

--------------------------------------------------------------------------------
-- SET LOGGER
--------------------------------------------------------------------------------

local logfile
if config.outdir ~= '' then
   paths.mkdir(config.outdir)
   logfile = io.open(paths.concat(config.outdir, 'log.txt'))
   print('Log file: ' .. paths.concat(config.outdir, 'log.txt'))
end

--------------------------------------------------------------------------------
-- LOAD DATA
--------------------------------------------------------------------------------

local trainfilename = paths.concat(config.data, 'train.txt')
local validfilename = paths.concat(config.data, 'valid.txt')
local testfilename  = paths.concat(config.data, 'test.txt')

local dic
if paths.filep(paths.concat(config.data, 'dic.txt')) then
   dic = data.loaddictionary(paths.concat(config.data, 'dic.txt'))
else
   dic = data.makedictionary(trainfilename)
   data.savedictionary(dic, paths.concat(config.data, 'dic.txt'))
end
dic = data.sortthresholddictionary(dic, config.threshold)
collectgarbage()
collectgarbage()

local ntoken = #dic.idx2word
local bsz    = config.batchsize
local tbsz   = config.testbatchsize
local bptt   = config.bptt

local batch = {
    train = data.loadfile(trainfilename, dic),
    valid = data.loadfile(validfilename, dic),
    test  = data.loadfile(testfilename,  dic),
}
collectgarbage()

local train = tnt.DatasetIterator(tnt.RNNDataset(batch.train, bsz,  bptt))
local valid = tnt.DatasetIterator(tnt.RNNDataset(batch.valid, tbsz, bptt))
local test  = tnt.DatasetIterator(tnt.RNNDataset(batch.test , tbsz, bptt))

--------------------------------------------------------------------------------
-- MAKE MODEL
--------------------------------------------------------------------------------

local initrange = config.initrange or 0.1

local lut = nn.LookupTable(ntoken, config.isz)
lut.weight:uniform(-initrange, initrange)
lut:cuda()

local rnn = nn.LSTM{
   inputsize = config.isz,
   hidsize   = config.nhid,
   nlayer    = config.nlayer,
   usecudnn  = config.usecudnn,
}

local cutoff = tablex.map(tonumber, stringx.split(config.cutoff, ','))
table.insert(cutoff, ntoken)

local decoder = nn.AdaptiveSoftMax(config.nhid, cutoff)
local crit = nn.AdaptiveLoss(cutoff)

onsample = function(state)
   state.inputlut = state.sample.input:cuda()
   lut:forward(state.inputlut)
   state.sample.input = {state.hid, lut.output}

   local target = state.sample.target:cuda()
   state.sample.target = target:view(target:nElement())

   decoder:setTarget(state.sample.target)
end

local model = nn.Sequential()
   :add(nn.ParallelTable()
           :add(nn.Identity())
           :add(nn.Sequential()
                   :add(nn.Dropout(config.dropout))
                   :add(nn.SplitTable(1))
               )
       )
   :add(rnn)
   :add(nn.SelectTable(2))
   :add(nn.SelectTable(-1))
   :add(nn.JoinTable(1))
   :add(nn.Dropout(config.dropout))
   :add(decoder)

collectgarbage()

model:cuda()
crit:cuda()

--------------------------------------------------------------------------------
-- TORCHNET
--------------------------------------------------------------------------------
local timer     = tnt.TimeMeter{unit = true}
local logtimer  = tnt.TimeMeter()
local tottimer  = tnt.TimeMeter()

local trainloss = tnt.AverageValueMeter()

local function runvalidation(network, criterion, iterator)
   local engine = tnt.SGDEngine()
   local meter  = tnt.AverageValueMeter()

   function engine.hooks.onStart(state)
      state.hid = rnn:initializeHidden(tbsz)
   end

   engine.hooks.onSample = onsample

   function engine.hooks.onForwardCriterion(state)
      meter:add(state.criterion.output)
      state.hid = tnt.utils.table.clone(rnn:getLastHidden())
   end

   engine:test{
      network   = network,
      iterator  = iterator,
      criterion = criterion,
   }

   return meter:value()
end

local engine = tnt.OptimEngine()

function engine.hooks.onStart(state)
   local eps = config.epsilon
   state.optim.paramVariance = state.gradParams:clone():fill(eps)
   state.optim.paramStd      = state.gradParams:clone()
   state.optim.lutVariance   = torch.Tensor(ntoken, 1):typeAs(lut.weight):fill(eps)
   state.hid = rnn:initializeHidden(bsz)
end

function engine.hooks.onStartEpoch(state)
    timer:reset()
    trainloss:reset()
end

engine.hooks.onSample = onsample

function engine.hooks.onBackward(state)
   -- clip gradients
   if config.clip > 0 then
      local norm = state.gradParams:norm()
      if norm > config.clip then
         state.gradParams:div(math.max(norm, 1e-6) / config.clip)
      end
   end

   local gradinput = model.gradInput[2]:view(bptt * bsz, config.isz)
   local idx       = state.inputlut:view(bptt * bsz)
   local variance  = state.optim.lutVariance
   variance:indexAdd(1, idx, torch.pow(gradinput, 2):mean(2))
   gradinput:cdiv(torch.sqrt(variance:index(1, idx):expandAs(gradinput)))
   lut:accUpdateGradParameters(state.inputlut, model.gradInput[2], state.config.learningRate)
end

function engine.hooks.onUpdate(state)
   trainloss:add(state.criterion.output)
   timer:incUnit()
   state.hid = tnt.utils.table.clone(rnn:getLastHidden())

   if logtimer:value() > 300 then
      local msbatch = timer:value() * 1000
      local trainppl = math.exp(trainloss:value())
      local validppl = math.exp(runvalidation(model, crit, valid))

      local str = string.format(
         '| epoch %2d | %8d samples | %7d ms/batch ' ..
         '| %5d min | train ppl %5.1f | valid ppl %5.1f',
         state.epoch, state.t, msbatch,
         tottimer:value() / 60, trainppl, validppl)
      print(str)
      if logfile then
         logfile:write(str .. '\n')
         logfile:flush()
      end

      collectgarbage()
      model:training()
      trainloss:reset()
      logtimer:reset()
      timer:reset()
   end
end

function engine.hooks.onEndEpoch(state)
   local msbatch = timer:value() * 1000
   local trainppl = math.exp(trainloss:value())
   local validppl = math.exp(runvalidation(model, crit, valid))
   local testppl  = math.exp(runvalidation(model, crit, test))

   local str = string.format(
      '| epoch %2d | %8d samples | %7d ms/batch | %5d min ' ..
      '| train ppl %5.1f | valid ppl %5.1f | test ppl %5.1f',
      state.epoch, state.t, msbatch, tottimer:value() / 60,
      trainppl, validppl, testppl)
   print(str)
   if logfile then
      logfile:write(str .. '\n')
      logfile:flush()
   end

   collectgarbage()
   model:training()
   trainloss:reset()
   logtimer:reset()
   timer:reset()

   if state.epoch >= 5 then
      state.config.learningRate = state.config.learningRate / 2
   end
end

tottimer:reset()
local config_opt = {
   learningRate = config.lr,
}

engine:train{
   network     = model,
   criterion   = crit,
   iterator    = train,
   optimMethod = optim.adagrad,
   maxepoch    = config.maxepoch,
   config      = config_opt,
}

--------------------------------------------------------------------------------
-- MODEL SAVING
--------------------------------------------------------------------------------

if config.outdir ~= '' then
   local model = nn.Sequential()
      :add(nn.ParallelTable()
              :add(nn.Identity())
              :add(nn.Sequential()
                      :add(lut)
                      :add(nn.Dropout(config.dropout))
                      :add(nn.SplitTable(1))
                  )
          )
      :add(rnn)
      :add(nn.SelectTable(2))
      :add(nn.SelectTable(-1))
      :add(nn.JoinTable(1))
      :add(nn.Dropout(config.dropout))
      :add(decoder)

   torch.save(paths.concat(config.outdir, 'model.t7'),
              {model = model, dic = dic, cutoff = cutoff, config = config})
end
