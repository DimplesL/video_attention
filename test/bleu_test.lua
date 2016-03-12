require 'torch'
require 'cutorch'
require 'bleu'

function idxTest(pred, truth, score)
	print 'Predicted: '
	print(pred)
	print 'Truth: '
	print(truth)
	print 'True BLEU:'
	print(score)
	print 'Computed BLEU: '
	computed = bleu.idxBleu(pred, truth)
	print(computed)
	if computed == score then
		print 'PASSED'
	else
		print 'FAILED'
		exit(1)
	end
end

print 'Computing BLEU with handmade tokens...'
idxTest({}, {{}}, 1)
idxTest({}, {{2}}, 1)
idxTest({}, {{5}}, 0)
idxTest({2}, {{5}}, 0)
idxTest({5}, {{5}}, 1)
idxTest({5}, {{6}}, 0)
idxTest({5}, {{6}}, 0)
idxTest({5, 6}, {{6}}, 0.5)
idxTest({5, 6}, {{6, 6}}, 0.5)
idxTest({5, 6, 6}, {{6}}, 1/3)
idxTest({6, 6, 6}, {{6}, {6}}, 1/3)
idxTest({5, 6, 7}, {{5}, {6, 7}}, 1)
idxTest({5, 6, 7}, {{5}, {6, 7}}, 1)
idxTest({5, 6, 7}, {{6}, {7}}, 2/3)

print 'Loading the torch model...'
local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/tiny_coco.h5')
cmd:option('-checkpoint','/data/checkpoints/64_1536_2_2500_10000.t7')
cmd:option('-device', 0)
local opt = cmd:parse(arg)

device = 1
cutorch.setDevice(device)
local checkpoint = torch.load(opt.checkpoint)

print 'Computing BLEU on the tiny dataset...'
local score = bleu.getScore(checkpoint, opt.h5, 'val', 'cuda', device, 32)
print(score)
if score >= 0 and score <= 1 then
	print 'PASSED'
else
	print 'FAILED'
	exit(1)
end

