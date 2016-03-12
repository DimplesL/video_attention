require 'torch'
require 'nn'

require 'TemporalAdapter'
require 'VanillaRNN'
require 'LSTM'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.rnns = {}
  self.net = nn.Sequential()

  self.lookup = nn.LookupTable(V, D)
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = false
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      self.net:add(nn.TemporalAdapter(nn.BatchNormalization(H)))
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  self.net:add(nn.TemporalAdapter(nn.Linear(H, V)))
end


function LM:updateOutput(input)
  -- unpack h0 and x
  local h0, x = nil, nil
  if torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false,"invalid input")
  end
  -- forward through the lookup layer
  local out = self.lookup:forward(x)
  -- package h0 with out from lookup
  self.rnn_input = {h0,out}
  -- call forward on rest of net
  return self.net:forward(self.rnn_input)
end


function LM:backward(input, gradOutput, scale)
  -- run backward through rnns, using saved output of lookup table
  local out = self.net:backward(self.rnn_input, gradOutput, scale)
  local grad_h0 = out[1]
  local grad_rnn = out[2]
  -- run backwards through lookup, using true input
  return self.lookup:backward(input, grad_rnn, scale)
end


function LM:parameters()
  local function tinsert(to, from)
    if type(from) == 'table' then
      for i=1,#from do
        tinsert(to,from[i])
      end
    else
      table.insert(to,from)
    end
  end
  local w = {}
  local gw = {}
  local mw,mgw = self.lookup:parameters()
  if mw then
    tinsert(w,mw)
    tinsert(gw,mgw)
  end
  for i=1,#self.net.modules do
    local mw,mgw = self.net.modules[i]:parameters()
    if mw then
      tinsert(w,mw)
      tinsert(gw,mgw)
    end
  end
  return w,gw
end

function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. " " .. token
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.

Inputs:
- init: String of length T0
- length: Number of characters to sample

Returns:
- sampled: (1, length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs)
  -- max length of caption
  local T = utils.get_kwarg(kwargs, 'length', 100)
  -- initial hidden state (image features)
  local h0 = utils.get_kwarg(kwargs, 'h0')
  local N,H = h0:size(1), h0:size(2)
  -- array holding sampled caption
  local sampled = torch.LongTensor(N, T)
  -- storage for scores and 
  local scores
  -- reset hidden and cell states
  self:resetStates()
  -- remember hidden and cell states between calls to forward
  for i, rnn in ipairs(self.rnns) do
    rnn:rememberStates(true)
  end
  -- get start token
  local x = torch.CudaTensor(N,1):fill(self.token_to_idx["<START>"])
  -- first forward pass
  scores = self:forward({h0,x})
  for t = 1, T do
    -- get the NxTxV (in this case 1x1xV) scores and take the argmax
    local _, next_word = scores:max(3)
    -- unpack the next word
    next_word = next_word[{{}, {}, 1}]
    -- copy the word into sampled
    sampled[{{}, {t, t}}]:copy(next_word)
    -- forward again with the sampled word
    scores = self:forward(next_word)
  end
  self:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:rememberStates(false)
  end
  return sampled
end


function LM:clearState()
  self.net:clearState()
end
