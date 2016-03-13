require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')

  self.split_sizes = {}
  -- Lovingly slurp all the moist, dripping data from the hard disk into memory
  -- SHARE THE LOAD
  local f = hdf5.open(h5_file, 'r')
  self.split_sizes['train'] = math.floor(f:read('/train_feats'):dataspaceSize()[1]/self.batch_size)
  self.split_sizes['val'] = math.floor(f:read('/val_feats'):dataspaceSize()[1]/self.batch_size)
  self.split_sizes['test'] = math.floor(f:read('/test_feats'):dataspaceSize()[1]/self.batch_size)
  self.feat_len = f:read('/train_feats'):dataspaceSize()[2]
  self.capt_len = f:read('/train_captions'):dataspaceSize()[2]
  self.split_idxs = {train=1, val=1, test=1}
  self.splits = {train={f:read('/train_feats'),f:read('/train_captions')},
                 val={f:read('/val_feats'),f:read('/val_captions')},
                 test={f:read('/test_feats')}}
end


function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local start_idx = self.batch_size*(idx-1)+1
  local end_idx = self.batch_size*idx
  x = self.splits[split][1]:partial({start_idx,end_idx},{1,self.feat_len})
  if split ~= 'test' then
    y = self.splits[split][2]:partial({start_idx,end_idx},{1,self.capt_len})
  end

  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end

