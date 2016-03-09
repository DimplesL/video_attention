require 'torch'
require 'cutorch'
require 'bleu'


print 'Loading the torch model...'
local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/validation_coco.h5')
cmd:option('-checkpoint','/data/checkpoints/64_1536_2_2500_10000.t7')
cmd:option('-device', 0)
local opt = cmd:parse(arg)

device = 2
cutorch.setDevice(device)
local checkpoint = torch.load(opt.checkpoint)

print 'Computing BLEU on the tiny dataset...'
local score = bleu.getScore(checkpoint, opt.h5, 'val', 'cuda', device)
print(score)
if score >= 0 and score <= 1 then
	print 'PASSED'
else
	print 'FAILED'
	exit(1)
end

