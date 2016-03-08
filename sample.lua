require 'torch'
require 'nn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
require 'cutorch'
require 'cunn'
cutorch.setDevice(opt.gpu + 1)
model:cuda()
print(string.format('Running with CUDA on GPU %d', opt.gpu))

model:evaluate()
local sample = model:sample(opt)
print(sample)
