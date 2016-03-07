require 'torch'

bleu = {}

bleu.mode = {
    "cpu",
    "cuda",
    "cl"
}

bleu.split = {
    "train",
    "val"
}

function bleu.getScore(checkpoint, h5name, split, mode, device)
-- Return the bleu score over the specified split.
-- 
-- Parameters:
--   checkpoint: The torch checkpoint
--   h5name: The name of the .h5 file containing the data
--   split: Either bleu.split.train or bleu.split.val
--   mode: Either bleu.cpu, bleu.cuda, or bleu.cl
--   device: The device number to use, e.g. 0 for GPU 0.
--
-- Returns a number giving the average bleu score per caption.

    -- Choose a torch datatype and computing backend
    local dtype = 'torch.FloatTensor'
    if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(device)
        dtype = 'torch.CudaTensor'
        print(string.format('Running BLEU with CUDA on GPU %d', device))   
    elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
        require 'cltorch'
        require 'clnn'
        cltorch.setDevice(device)
        dtype = torch.Tensor():cl():type()
        print(string.format('Running BLEU with OpenCL on GPU %d', device))  
    else
        -- Memory benchmarking is only supported in CUDA mode
        print 'Running BLEU in CPU mode'
    end

    -- Load the checkpoint and model
    local checkpoint = torch.load(opt.checkpoint)
    local model = checkpoint.model
    model:type(dtype)
    local crit = nn.CrossEntropyCriterion():type(dtype)

    -- Mangle some strings
    local map_dset_name = split + '_map'
    local feat_dset_name = split + '_feats'
    local captions_dset_name = split + '_captions'

    -- Open the h5 file and datasets
    local f = hdf5.open(h5_file, 'r')
    local map_dset = f:read(map_dset_name) 
    local feat_dset = f:read(feat_dset_name)
    local captions_dset = f:read(captions_dset_name)

    -- Get the number of images, feature size, etc.
    local num_imgs = map_dset:dataspaceSize()[1]
    local idxs_per_img = map_dset:dataspaceSize()[2]
    local feat_len = feat_dset:dataspaceSize()[2]
    local capt_len = captions_dset:dataspaceSize()[2]

    -- Initialize the model
    model:evaluate()
    model:resetStates()

    -- Process each image
    local score = 0
    for i=1,num_imgs do 
        -- Print our progress
        if i % 100 = 0 do
            print(string.format('Computing BLEU for image %d'), i)
        end

        -- Load the caption indices for this image
        local capt_idxs = map_dset:partial({i, i}, {1, idxs_per_img})

        -- Load the features and convert to Torch's format
        local feat_idx = capt_idxs[0] + 1
        local x = feat_dset:partial({feat_idx, feat_idx},{1, self.feat_len})
        x = x:type(dtype)

        -- Forward pass
        local vocab_scores = model:forward(x):view(N * T, -1)

        -- Convert the scores to words
        --TODO

        -- Accumulate the BLEUscore
        score = score + idxBleu(pred, truth)
    end

    return score / num_imgs

function idxBleu(pred, truth)
--TODO
