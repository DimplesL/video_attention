require "AttentionLSTM"
--require "cutorch"
--require "cunn"
--cutorch.setDevice(3)
repl = require 'trepl'
ID =11
IH=10
IW=10
H=9
V=8
T=7
N=6
D = ID + V
a = nn.AttentionLSTM({ID,IH,IW},V,H)
seq = torch.rand(N,T,V)
ims = torch.rand(N,ID,IH,IW)
h0 = torch.rand(N,H)
a0 = torch.rand(N,IH*IW)
c0 = torch.rand(N,H)
old_ims = torch.Tensor()
old_ims:resizeAs(ims):copy(ims)
old_seq = torch.Tensor()
old_seq:resizeAs(seq):copy(seq)
out = a:forward({c0,h0,a0,ims,seq})
gout = torch.Tensor()
gout:resizeAs(out):fill(1)
grad = a:backward({c0,h0,a0,ims,seq},gout,1)
repl()
