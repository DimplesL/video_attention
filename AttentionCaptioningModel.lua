require 'torch'
require 'nn'

require 'TemporalAdapter'
require 'AttentionLSTM'

local utils = require 'util.utils'


local AM, parent = torch.class('nn.AttentionCaptioningModel', 'nn.Module')


function AM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.im_size = utils.get_kwarg(kwargs, 'im_size')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size
  local ID, IH, IW = self.im_size[1], self.im_size[2], self.im_size[3]

  self.lookup = nn.LookupTable(V, D)
  self.net = nn.Sequential()
  self.rnns = {nn.AttentionLSTM(self.im_size,D,H)}
  self.net:add(self.rnns[1])
  self.net:add(nn.TemporalAdapter(nn.Linear(H, V)))
end


function AM:updateOutput(input)
  -- unpack h0 and x
  local h0, I, x = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    h0, I, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    I, x = unpack(input)
  else
    assert(false,"invalid input")
  end
  -- forward through the lookup layer
  local w_out = self.lookup:forward(x)
  -- package h0 with out from lookup
  self.rnn_input = {h0, I, w_out}
  -- call forward on rest of net
  return  self.net:forward(self.rnn_input)
end


function AM:backward(input, gradOutput, scale)
  local nout = self.net:backward(self.rnn_input, gradOutput, scale)
  local grad_h0 = nout[1]
  local grad_I = nout[2]
  local grad_x = nout[3]
  -- run backwards through lookup, using true input
  return self.lookup:backward(input, grad_x, scale)
end


function AM:parameters()
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

function AM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function AM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function AM:decode_string(encoded)
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
function AM:sample(kwargs)
  -- max length of caption
  local T = utils.get_kwarg(kwargs, 'length', 100)
  -- initial hidden state (image features)
  local h0 = utils.get_kwarg(kwargs, 'h0', torch.zeros(self.rnn_size))
  -- array holding sampled caption
  local sampled = torch.LongTensor(1, T)
  -- storage for scores and 
  local scores
  -- reset hidden and cell states
  self:resetStates()
  -- remember hidden and cell states between calls to forward
  for i, rnn in ipairs(self.rnns) do
    rnn:rememberStates(true)
  end
  -- get start token
  local x = torch.LongTensor(1):fill(self.token_to_idx["<START>"]):view(1,-1)
  -- first forward pass
  scores = self:forward({h0,x})[{{}, {1, 1}}]
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
  return sampled[1]
end


function AM:clearState()
  self.net:clearState()
end
