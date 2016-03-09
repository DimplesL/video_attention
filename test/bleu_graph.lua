require 'torch'
require 'cutorch'
require 'bleu'


print 'Loading the torch model...'
local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/coco_validation.h5')
cmd:option('-checkpoint','/data/checkpoints/64_1536_2_2500_1000.t7')
cmd:option('-device', 1)
cmd:option('-split', 'train')
cmd:option('-batchSize', 128)
local opt = cmd:parse(arg)

print('Loading checkpoint'..opt.checkpoint)
cutorch.setDevice(opt.device)
local checkpoint = torch.load(opt.checkpoint)

print('Running split '..opt.split)
local score = bleu.getScore(checkpoint, opt.h5, opt.split, 'cuda', opt.device)
print(score)
