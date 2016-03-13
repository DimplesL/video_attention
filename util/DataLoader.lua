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

  -- Read the h5 file and datasets
  local f = hdf5.open(h5_file, 'r')
  local train_feats = f:read('/train_feats')
  local val_feats = f:read('/val_feats')
  local test_feats = f:read('/test_feats')
  local train_captions = f:read('/train_captions')
  local val_captions = f:read('/val_captions')
 
  -- Set up the parameters 
  self.split_sizes['train'] = math.floor(train_feats:dataspaceSize()[1]/self.batch_size)
  self.split_sizes['val'] = math.floor(val_feats:dataspaceSize()[1]/self.batch_size)
  self.split_sizes['test'] = math.floor(test_feats:dataspaceSize()[1]/self.batch_size)
  self.capt_len = train_captions:dataspaceSize()[2]
  self.split_idxs = {train=1, val=1, test=1}
  self.splits = {train={train_feats,train_captions},
                 val={val_feats,val_captions},
                 test={test_feats}}

 -- Set up the image feature size
 self.image_size = nil
 self.feat_len = train_feats:dataspaceSize()[2]
 if train_feats:dataspaceSize()[3] ~= nil then
   self.image_size = {}
   for idx, size in pairs(train_feats:dataspaceSize()) do
     if idx > 1 then
       self.image_size[#self.image_size + 1] = size
     end
     if idx > 2 then
       self.feat_len = self.feat_len * size
     end
   end
 end
end


function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local start_idx = self.batch_size*(idx-1)+1
  local end_idx = self.batch_size*idx
  local x, y = nil, nil
  if self.image_size == nil then
    x = self.splits[split][1]:partial({start_idx,end_idx},{1,self.feat_len})
  else
    x = self.splits[split][1]:partial({start_idx,end_idx},
          {1,self.image_size[1]},{1,self.image_size[2]},{1,self.image_size[3]})
  end
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

