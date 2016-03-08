require 'torch'
require 'cutorch'
require 'bleu'

local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/tiny_coco2.h5')
cmd:option('-checkpoint', '/data/checkpoints/checkpoint_7750.t7')
cmd:option('-device', 0)
local opt = cmd:parse(arg)

checkpoint = torch.load(opt.checkpoint)

print(bleu.getScore(checkpoint, opt.h5, 'val', 'cuda', 1))
