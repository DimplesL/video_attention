require 'torch'
require 'nn'


local layer, parent = torch.class('nn.AttentionLSTM', 'nn.Module')

-- Constructs an LSTM with spatial attention.
-- the spatial attention mechanism computes a set of weights at each timestep
-- that are used to average the input feature volume
-- the lstm accept a DxD1xD2 tensor and uses its D1xD2 weights
-- to take a weighted average of the volume, producing a vector of length
-- D.
--
-- These weights are called a, and they are computed as a function of
-- the hidden state and the input
--
-- So we exted weight, adding IHxIw rows (also add to bias) 
function layer:__init(image_input_dims, seq_input_dim, hidden_dim)
  parent.__init(self)
  local ID,IH,IW = unpack(image_input_dims) -- image input
  local SD = seq_input_dim --sequence input
  local H = hidden_dim -- hidden
  local D = SD + ID -- concatenate sequence and image to get full input
  self.image_input_dims = image_input_dims
  self.seq_input_dim = SD
  self.input_dim = D
  self.hidden_dim = H
  self.att_dim = {IH,IW}
  self.weight = torch.Tensor(D + H, (4 * H)+(IH*IW))
  self.gradWeight = torch.Tensor(D + H, (4 * H)+(IH*IW)):zero()
  self.bias = torch.Tensor((4 * H)+ (IH*IW))
  self.gradBias = torch.Tensor((4 * H)+(IH*IW)):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, T, H)
  self.gates = torch.Tensor()   -- This will be (N, T, 4H)
  self.input_buf = torch.Tensor()   -- This will be (N,D)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (H,)
  self.buffer4 = torch.Tensor() -- This will be (H,)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.att0 = torch.Tensor()
  self.remember_states = false

  self.grad_c0 = torch.Tensor()
  self.grad_h0 = torch.Tensor()
  self.grad_a0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.grad_seq = torch.Tensor()
  self.grad_im = torch.Tensor()
  self.grad_att = torch.Tensor()
  self.gradInput = {self.grad_c0, self.grad_h0, self.grad_a0, self.grad_att, self.grad_seq, self.grad_im}
 
  self.I = nil -- Remember the last image in remember_states mode
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  -- initialize all gates to 0 except forget gate
  self.bias:zero()
  self.bias[{{self.hidden_dim + 1, 2 * self.hidden_dim}}]:fill(1)
  self.weight:normal(0, std)
  return self
end

function layer:rememberStates(val)
  self.remember_states = val
end

function layer:resetStates()
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end

local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function layer:_unpack_input(input)
  local c0, h0, a0, I, x  = nil, nil, nil,nil,nil
  if torch.type(input) == 'table' and #input == 5 then
    c0, h0, a0, I, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 4 then
    h0, a0, I, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 3 then
    h0,I,x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    I,x = unpack(input)
  else
    assert(false, 'invalid input')
  end
  return c0, h0, a0, I, x
end


function layer:_get_sizes(input, gradOutput)
  local c0, h0, a0, I, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local SD = self.seq_input_dim
  local ID,IH,IW = unpack(self.image_input_dims)
  local H = self.hidden_dim
  check_dims(x,{N,T,SD})
  check_dims(I, {N,ID,IH,IW})
  if h0 then
    check_dims(h0, {N, H})
  end
  if c0 then
    check_dims(c0, {N, H})
  end
  if a0 then
    check_dims(a0,{N, IH*IW})
  end
  if gradOutput then
    check_dims(gradOutput, {N, T, H})
  end
  return N, T, SD, ID, IH, IW, H
end


function layer:softmax_forward_all(input)
  local sm = input:clone():zero()
  for i = 1,input:size(1) do
    sm[{i,{}}] = self:softmax_forward(input[{i,{}}])
  end
  return sm
end

function layer:softmax_backward_all(sm,dout)
  local dsm = sm:clone():zero()
  for i = 1,sm:size(1) do
    dsm[{i,{}}] = self:softmax_backward(sm[{i,{}}],dout[{i,{}}])
  end
  return dsm
end


function layer:softmax_forward(input)
  local x = input - input:max()
  local z = torch.sum(torch.exp(x))
  return torch.exp(x)/z
end

function layer:genDiag(v)
--diag that works for cuda....
   n = v:size(1)
   D = v:clone():resize(n, n):zero()
   for i = 1,n do
       D[{i,i}]= v[i]
   end
   return D
end

function layer:softmax_backward(sm,dout)
  local P = torch.ger(-sm,sm)
  P:add(layer:genDiag(sm))
  return torch.mv(P,dout)
end

--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- a0: Initial attention state, (N, IH*IW)
- I: Input images , (N,ID, IH, IW)
- x: Input sequence, (N, T, SD)

Output:
- h: Sequence of hidden states, (N, T, H)
--]]

function isnan(inp)
  return torch.any(inp:ne(inp))
end

function layer:updateOutput(input)
  local c0, h0, a0, I, x = self:_unpack_input(input)
  local N, T, SD, ID, IH, IW, H = self:_get_sizes(input)
  local D = SD + ID

  -- remember the previous I in remember_states mode
  if self.remember_states then
  	if not I then
	    I = self.I
  	else
	    self.I = I 
  	end
  end
  -- reshape image feature volume as matrix
  I = I:reshape(N,ID,IH*IW) -- I is the image features
  self._return_grad_c0 = (c0 ~= nil)
  self._return_grad_h0 = (h0 ~= nil)
  self._return_grad_a0 = (a0 ~= nil)
  if not c0 then -- if c0 is not provided
    c0 = self.c0 -- load from current
    -- if current c0 is not initialized or we aren't remembering states 
    -- the zero it out
    if c0:nElement() == 0 or not self.remember_states then
      c0:resize(N, H):zero()
    -- otherwise load previous c
    elseif self.remember_states then
      local prev_N, prev_T = self.cell:size(1), self.cell:size(2)
      assert(prev_N == N, 'batch sizes must be constant to remember states')
      c0:copy(self.cell[{{}, prev_T}])
    end
  end
  -- do the same for h 
  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.output:size(1), self.output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self.output[{{}, prev_T}])
    end
  end
  -- do the same for a,defaulting to uniform attention
  if not a0 then
    a0 = self.att0
    if a0:nElement() == 0 then
      a0:resize(N,IH*IW):fill(1/(IH*IW))
    end
  end
  -- create N copies of the bias
  local weight_height = (4*H) + (IH*IW)
  local bias_expand = self.bias:view(1, weight_height ):expand(N, weight_height)
  -- break apart the weight matrix into the part that multiplies x and the
  -- part that multiplies y
  local Wx = self.weight[{{1, D}}] -- is now Dx(4*H+IW*IH)
  local Wh = self.weight[{{D + 1, D + H}}] --is now Hx(4*H+IW*IH)

  -- set up buffers for h and c
  local h, c = self.output, self.cell
  local input_buf = self.input_buf
  h:resize(N, T, H):zero()
  c:resize(N, T, H):zero()
  input_buf:resize(N,T,D):zero()
  -- set up prev_h, prev_c
  local prev_h, prev_c, prev_a = h0, c0, a0
  -- holds the output of each gate
  self.gates:resize(N, T, weight_height):zero()
  for t = 1, T do
    --print('t: h,c,a:',t, isnan(prev_h), isnan(prev_c), isnan(prev_a))
    local next_h = h[{{}, t}] -- NxH
    local next_c = c[{{}, t}] -- NxH
    local cur_gates = self.gates[{{}, t}] -- current gate values
    local cur_input = input_buf[{{}, t}] -- current gate values
    cur_input[{{},{1,SD}}]:copy(x[{{},t}])-- put the current x into the input
    local tmp = torch.bmm(I,prev_a:reshape(N,IH*IW,1)) -- add the image to the input
    cur_input[{{},{SD+1,-1}}]:copy(tmp:squeeze())
    --cur_input[{{},{SD+1,-1}}]:view(N,ID,1):baddbmm(I:view(N,ID,IH*IW),prev_a:view(N,IH*IW,1)) -- add the image to the input
    cur_gates:addmm(bias_expand, cur_input:squeeze(), Wx) -- multiply NxD cur_input by Dx(...) Wx
    cur_gates:addmm(prev_h, Wh) -- multiply NxH prev_h by Hx(...) Wh
    cur_gates[{{}, {1, 3 * H}}]:sigmoid() -- take sigmoid of first 3 gates
    cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh() -- take tanh of last gate
    cur_gates[{{}, {4*H+1, -1}}] = self:softmax_forward_all(cur_gates[{{},{4*H+1,-1}}]) -- take softmax of attention
    local next_a = cur_gates[{{}, {4 * H + 1,-1}}]
    -- break out each of the gates
    local i = cur_gates[{{}, {1, H}}]
    local f = cur_gates[{{}, {H + 1, 2 * H}}]
    local o = cur_gates[{{}, {2 * H + 1, 3 * H}}]
    local g = cur_gates[{{}, {3 * H + 1, 4 * H}}]
    next_h:cmul(i, g) -- compute next hidden
    next_c:cmul(f, prev_c):add(next_h) -- compute next cell
    next_h:tanh(next_c):cmul(o) --finish computing hidden
    prev_h, prev_c, prev_a =  next_h, next_c, next_a --update vars
  end
  return self.output
end


function layer:backward(input, gradOutput, scale)
  scale = scale or 1.0
  local c0, h0, att0, I, x = self:_unpack_input(input)
  local N, T, SD, ID, IH, IW, H = self:_get_sizes(input)
  local D = SD + ID
  I = I:reshape(N,ID,IH*IW)
  if not c0 then c0 = self.c0 end
  if not h0 then h0 = self.h0 end
  if not att0 then att0 = self.att0 end -- this is initialized to the default uniform distribution in "updateOutput"

  local grad_c0, grad_h0, grad_a0 = self.grad_c0, self.grad_h0, self.grad_a0;
  local grad_x, grad_att, grad_im =  self.grad_x, self.grad_att, self.grad_im
  local h, c = self.output, self.cell
  local grad_h = gradOutput -- grad with respect to output

  local Wx = self.weight[{{1, D}}] -- left D cols of W
  local Wh = self.weight[{{D + 1, D + H}}] -- right D+H cols of W
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias
  grad_h0:resizeAs(h0):zero() 
  grad_c0:resizeAs(c0):zero()
  grad_a0:resize(N,IH*IW):zero()
  grad_x:resize(N,T,D):zero()
  grad_im:resizeAs(I):zero()
  grad_att:resizeAs(att0):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local grad_next_c = self.buffer2:resizeAs(c0):zero()
  local grad_next_att = self.buffer4:resizeAs(att0):zero()
  for t = T, 1, -1 do
    local next_h, next_c = h[{{}, t}], c[{{}, t}]
    local prev_h, prev_c, prev_att  = nil, nil, nil
    if t == 1 then
      prev_h, prev_c, prev_att = h0, c0, att0
    else
      prev_h, prev_c, prev_att = h[{{}, t - 1}], c[{{}, t - 1}], self.gates[{{}, t-1, {4 * H + 1, -1}}]
    end
    grad_next_h:add(grad_h[{{}, t}]) -- why???

    local i = self.gates[{{}, t, {1, H}}]
    local f = self.gates[{{}, t, {H + 1, 2 * H}}]
    local o = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
    local g = self.gates[{{}, t, {3 * H + 1, 4 * H}}]
    local next_att = self.gates[{{}, t, {4 * H + 1, -1}}]
    
    local grad_a = self.grad_a_buffer:resize(N, 4 * H + IH*IW):zero()
    local grad_ai = grad_a[{{}, {1, H}}]
    local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
    local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
    local grad_a_att = grad_a[{{}, {4 * H + 1, -1}}]

    
    -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
    -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
    -- to compute grad_ao; the other values can be overwritten after we compute
    -- grad_next_c
    local tanh_next_c = grad_ai:tanh(next_c)
    local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
    local my_grad_next_c = grad_ao
    -- this is weird. what it does is puts 1s in my_grad_next c
    -- and then subtracts tanh_next_c2 from those ones, then
    -- elementwise multiplies by o and grad_next_h
    -- h_t = o*tanh(c_t)
    -- dc_t = o*(1-tanh(c_t)^2)*dh_t
    my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
    grad_next_c:add(my_grad_next_c)
    
    -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
    -- that we can overwrite it.
    -- h_t = sigm(ao)*tanh(c_t)
    -- dao = sigm(ao)*(1-sigm(ao))*tanh(c_t)*dh_t
    grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

    -- Use grad_ai as a temporary buffer for computing grad_ag
    -- ct = f*c_{t-1} + i*tanh(ag)
    -- dg = dct*i*(1-tanh(ag)^2)
    local g2 = grad_ai:cmul(g, g) -- put g^2 in g2
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

    -- We don't need any temporary storage for these so do them last
    -- ct = f*c_{t-1} + sigm(ai)*g
    -- dai = dct*g*(sigm(ai)(1-sigm(ai))
    grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
    -- daf = dct*c_{t-1}*(sigm(af)(1-sigm(af))
    grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
    -- add grad_a_att computation that backpropogates gradient from att_softmax through softmax 
    --grad_a_att:add(grad_next_att:cmul(self.softmax:backward(next_att_presoftmax,grad_next_att)))
    grad_a_att:add(self:softmax_backward_all(next_att,grad_next_att))
    grad_x[{{}, t}]:mm(grad_a, Wx:t())
    -- split gradx into its image part and its 
    local grad_attended_im = grad_x[{{},t,{SD+1,-1}}]
    grad_Wx:addmm(scale, self.input_buf[{{}, t}]:t(), grad_a)
    grad_Wh:addmm(scale, prev_h:t(), grad_a)
    local grad_a_sum = self.buffer3:resize(H):sum(grad_a, 1)
    grad_b:add(scale, grad_a_sum)
    grad_next_att:zero()
    --grad_next_att:view(N,IH*IW,1) = torch.bmm(I:transpose(2,3),grad_attended_im:reshape(N,ID,1))
    grad_next_att = torch.bmm(I:transpose(2,3),grad_attended_im:reshape(N,ID,1)):squeeze()
    local tmp = torch.bmm(grad_attended_im:reshape(N,ID,1),prev_att:reshape(N,1,IH*IW))
    grad_im:add(tmp:view(N,ID,IH,IW))
    grad_next_h:mm(grad_a, Wh:t())
    grad_next_c:cmul(f)
  end
  grad_h0:copy(grad_next_h)
  grad_c0:copy(grad_next_c)
  grad_a0:copy(grad_next_att)
  self.grad_seq:resizeAs(x)
  self.grad_seq:copy(grad_x[{{},{},{1,SD}}])
  if self._return_grad_c0 and self._return_grad_h0 and self._return_grad_a0 then
    self.gradInput = {self.grad_c0, self.grad_h0, self.grad_a0, self.grad_im, self.grad_seq}
  elseif self._return_grad_h0 and self._return_grad_c0 then
    self.gradInput = {self.grad_h0, self.grad_c0, self.grad_im, self.grad_seq}
  elseif self._return_grad_h0 and self._return_grad_a0 then
    self.gradInput = {self.grad_h0, self.grad_a0, self.grad_im, self.grad_seq}
  elseif self._return_grad_c0 and self._return_grad_a0 then
    self.gradInput = {self.grad_c0, self.grad_a0, self.grad_im, self.grad_seq}
  elseif self._return_grad_c0 then
    self.gradInput = {self.grad_c0, self.grad_im, self.grad_seq}
  elseif self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_im, self.grad_seq}
  elseif self._return_grad_a0 then
    self.gradInput = {self.grad_a0, self.grad_im, self.grad_seq}
  else
    self.gradInput = {self.grad_im, self.grad_seq}
  end

  return self.gradInput
end


function layer:clearState()
  self.cell:set()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.buffer4:set()
  self.grad_a_buffer:set()

  self.grad_c0:set()
  self.grad_h0:set()
  self.grad_a0:set()
  self.grad_x:set()
  self.grad_seq:set()
  self.grad_im:set()
  self.grad_att:set()
  self.output:set()
  self.input_buf:set()
end


function layer:updateGradInput(input, gradOutput)
  self:backward(input, gradOutput, 0)
end


function layer:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end
