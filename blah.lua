require "AttentionLSTM"
--require "cutorch"
--require "cunn"
--cutorch.setDevice(3)
repl = require 'trepl'
ID =2048
IH=8
IW=8
H=1024
V=1024
N=32
T=50
D = ID + V
a = nn.AttentionLSTM({ID,IH,IW},V,H)
seq = torch.rand(N,T,V)
ims = torch.rand(N,ID,IH,IW)
out = a:forward({ims,seq})
gout = torch.Tensor()
gout:resizeAs(out):fill(1)
grad = a:backward({ims,seq},gout,1)
repl()
