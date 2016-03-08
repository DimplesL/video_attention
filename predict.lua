require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
np = require 'npy4th'
require 'LanguageModel'
local utils = require 'util.utils'

local cmd = torch.CmdLine()

-- options
cmd:option('-images', '')
cmd:option('-checkpoint', '')
cmd:option('-input_json', 'data/tiny-shakespeare.json')
cmd:option('-sample_length', 50)
cmd:option('-gpu', 0)

local opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu + 1)
print(string.format('Running with CUDA on GPU %d', opt.gpu))

-- Initialize the DataLoader and vocabulary
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
local token_to_idx = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
  token_to_idx[v] = tonumber(k)
end

checkpoint = torch.load(opt.checkpoint)
model = checkpoint.model

for line in io.lines(opt.images) do
  print(line)
  local feat = np.loadnpy(line):type('torch.CudaTensor')
  feat = feat:mean(2):mean(3):reshape(1,2048)
  sample = model:sample({length=opt.sample_length,h0=feat})
  print(sample)
end
