require 'torch'
require 'nn'

require 'AttentionLSTM'
local gradcheck = require 'util.gradcheck'


local tests = {}
local tester = torch.Tester()


local function check_size(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end

function tests.forward()
-- Test that we can run a forward pass
  local N, T, SD, H, ID, IH,IW = 5, 6, 7, 8, 9, 10,10 

  local x = torch.randn(N, T, SD)
  local I = torch.randn(N, ID, IH, IW)
  local h0 = torch.randn(N, H)
  local c0 = torch.randn(N, H)
  local a0 = torch.randn(N, IH*IW)
  
  local lstm = nn.AttentionLSTM({ID,IH,IW}, SD, H)
  local h = lstm:forward({c0, h0, a0, I, x})

  tester:asserteq(h, h)
end

function tests.gradcheck()
  local N, T, SD, H, ID, IH,IW = 5, 6, 7, 8, 9, 10,10 

  local x = torch.randn(N, T, SD)
  local I = torch.randn(N, ID, IH, IW)
  local h0 = torch.randn(N, H)
  local c0 = torch.randn(N, H)
  local a0 = torch.randn(N, IH*IW)
  
  local lstm = nn.AttentionLSTM({ID,IH,IW}, SD, H)
  local h = lstm:forward({c0, h0, a0, I, x})

  local dh = torch.randn(#h)

  lstm:zeroGradParameters()
  local dc0, dh0, da0, dI, dx = unpack(lstm:backward({c0, h0, a0, I, x}, dh))
  local dw = lstm.gradWeight:clone()
  local db = lstm.gradBias:clone()

  local function fh0(h0) return lstm:forward{c0, h0, a0, I, x} end
  local function fc0(c0) return lstm:forward{c0, h0, a0, I, x} end
  local function fa0(a0) return lstm:forward{c0, h0, a0, I, x} end
  local function fx(x)   return lstm:forward{c0, h0, a0, I, x} end
  --local function fI(I)   return lstm:forward{c0, h0, a0, I, x} end

  local function fw(w)
    local old_w = lstm.weight
    lstm.weight = w
    local out = lstm:forward{c0, h0, a0, I, x}
    lstm.weight = old_w
    return out
  end

  local function fb(b)
    local old_b = lstm.bias
    lstm.bias = b
    local out = lstm:forward{c0, h0, a0, I, x}
    lstm.bias = old_b
    return out
  end
  
  local dx_num = gradcheck.numeric_gradient(fx, x, dh)
   
  out = fx(x)
  if torch.any(out:ne(out)) then
    print(out)
    print(out:ne(out))
    print("This is fucked, boys")
  end

  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
  local dc0_num = gradcheck.numeric_gradient(fc0, c0, dh)
  local da0_num = gradcheck.numeric_gradient(fa0, a0, dh)
--  local dI_num = gradcheck.numeric_gradient(fI, I, dh)
  local dw_num = gradcheck.numeric_gradient(fw, lstm.weight, dh)
  local db_num = gradcheck.numeric_gradient(fb, lstm.bias, dh)

  local dx_error = gradcheck.relative_error(dx_num, dx)
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)
  local dc0_error = gradcheck.relative_error(dc0_num, dc0)
  local da0_error = gradcheck.relative_error(da0_num, da0)
  local dw_error = gradcheck.relative_error(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  tester:assertle(dc0_error, 1e-5)
  tester:assertle(da0_error, 1e-5)
  tester:assertle(dx_error, 1e-5)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-5)
end

--[[
-- Make sure that everything works correctly when we don't pass an initial cell
-- state; in this case we do pass an initial hidden state and an input sequence
function tests.noCellTest()
  local N, T, D, H, I = 5, 6, 7, 8, {2048, 8, 8}
  local lstm = nn.AttentionLSTM(I, D, H)

  for t = 1, 3 do
    local x = torch.randn(N, T, D)
    local h0 = torch.randn(N, H)
    local dout = torch.randn(N, T, H)

    local out = lstm:forward{h0, x}
    local din = lstm:backward({h0, x}, dout)

    tester:assert(torch.type(din) == 'table')
    tester:assert(#din == 2)
    check_size(din[1], {N, H})
    check_size(din[2], {N, T, D})

    -- Make sure the initial cell state got reset to zero
    tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
  end
end


-- Make sure that everything works when we don't pass initial hidden or initial
-- cell state; in this case we only pass input sequence of vectors
function tests.noHiddenTest()
  local N, T, D, H, I = 5, 6, 7, 8, {2048, 8, 8}
  local lstm = nn.AttentionLSTM(I, D, H)

  for t = 1, 3 do
    local x = {torch.Tensor(N, I[1], I[2], I[3]), torch.randn(N, T, D)}
    local dout = torch.randn(N, T, H)

    local out = lstm:forward(x)
    local din = lstm:backward(x, dout)

    tester:assert(torch.isTensor(din))
    check_size(din, {N, T, D})

    -- Make sure the initial cell state and initial hidden state are zero
    tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
    tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
  end
end


function tests.rememberStatesTest()
  local N, T, D, H, I = 5, 6, 7, 8, {2048, 8, 8}
  local lstm = nn.AttentionLSTM(I, D, H)
  lstm.remember_states = true

  local final_h, final_c = nil, nil
  for t = 1, 4 do
    local x = {torch.Tensor(N, I[1], I[2], I[3]), torch.randn(N, T, D)}
    local dout = torch.randn(N, T, H)
    local out = lstm:forward(x)
    local din = lstm:backward(x, dout)

    if t == 1 then
      tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
      tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
    elseif t > 1 then
      tester:assertTensorEq(lstm.c0, final_c, 0)
      tester:assertTensorEq(lstm.h0, final_h, 0)
    end
    final_c = lstm.cell[{{}, T}]:clone()
    final_h = out[{{}, T}]:clone()
  end

  -- Initial states should reset to zero after we call resetStates
  lstm:resetStates()
  local x = {torch.Tensor(N, I[1], I[2], I[3]), torch.randn(N, T, D)}
  local dout = torch.randn(N, T, H)
  lstm:forward(x)
  lstm:backward(x, dout)
  tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
  tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
end
--]]

tester:add(tests)
tester:run()

