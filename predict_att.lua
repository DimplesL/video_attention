require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
np = require 'npy4th'
--require 'image'
require 'AttentionCaptioningModel'
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
  local feat1 = np.loadnpy(line):type('torch.CudaTensor')
  ID,IH,IW = feat1:size(1),feat1:size(2),feat1:size(3)
  feat1  = feat1:resize(1,ID,IH,IW)
  local feat = torch.cat(feat1,torch.zeros(1,ID,IH,IW):type('torch.CudaTensor'),1)
  local sample, att = model:sample({length=opt.sample_length,I=feat})
  att = att[1]:reshape(opt.sample_length+1,IH,IW)
  np.savenpy('att_ims/'..string.sub(line,35,46)..'.npy', att)
  print(line)
  print(string.sub(line,35,46)..": "..model:decode_string(sample[1]))
end
