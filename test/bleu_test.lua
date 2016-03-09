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
idxTest({}, {{1}}, 1)
idxTest({}, {{4}}, 0)
idxTest({1}, {{4}}, 0)
idxTest({4}, {{4}}, 1)
idxTest({4}, {{5}}, 0)
idxTest({4}, {{5}}, 0)
idxTest({4, 5}, {{5}}, 0.5)
idxTest({4, 5}, {{5, 5}}, 0.5)
idxTest({4, 5, 5}, {{5}}, 1/3)
idxTest({4, 5, 5}, {{5}, {5}}, 1/3)
idxTest({4, 5, 6}, {{4}, {5, 6}}, 1)
idxTest({4, 5, 6}, {{4}, {5, 6}}, 1)
idxTest({4, 5, 6}, {{5}, {6}}, 2/3)

print 'Loading the torch model...'
local cmd = torch.CmdLine()
cmd:option('-h5', '/data/coco/tiny_coco2.h5')
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

