require 'torch'
require 'cutorch'
require 'bleu'


print 'Loading the torch model...'
local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/coco_bgr.h5')
cmd:option('-checkpoint','/data/checkpoints/64_1536_2_2500_10000.t7')
cmd:option('-split', 'val')
cmd:option('-device', 1)
cmd:option('-batchSize', 512)
local opt = cmd:parse(arg)

cutorch.setDevice(opt.device)
local checkpoint = torch.load(opt.checkpoint)

print 'Computing BLEU on the tiny dataset...'
local score = bleu.getScore(checkpoint, opt.h5, opt.split, 'cuda', opt.device, opt.batchSize)
print(score)

