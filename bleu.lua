require 'torch'
require 'LanguageModel'
require 'hdf5'

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
--   device: The device number to use, e.g. 1 for GPU 1.
--
-- Returns a number giving the average bleu score per caption.

    -- Choose a torch datatype and computing backend
    local dtype = 'torch.FloatTensor'
    if mode == 'cuda' then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(device)
        dtype = 'torch.CudaTensor'
        print(string.format('Running BLEU with CUDA on GPU %d', device))   
    elseif mode == 'cl' then
        require 'cltorch'
        require 'clnn'
        cltorch.setDevice(device)
        dtype = torch.Tensor():cl():type()
        print(string.format('Running BLEU with OpenCL on GPU %d', device))  
    else
        -- Memory benchmarking is only supported in CUDA mode
        print 'Running BLEU in CPU mode'
    end

    -- Convert the model to the appropriate type
    local model = checkpoint.model:type(dtype)
    local crit = nn.CrossEntropyCriterion():type(dtype)

    -- Mangle some strings
    local map_dset_name = split..'_map'
    local feat_dset_name = split..'_feats'
    local captions_dset_name = split..'_captions'

    -- Open the h5 file and datasets
    local f = hdf5.open(h5name, 'r')
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
        if i % 100 == 0 then
            print(string.format('Computing BLEU for image %d'), i)
        end

        -- Load the caption indices for this image
        local capt_idxs = torch.totable(map_dset:partial({i, i}, {1, idxs_per_img}))[1]

        -- Load the features and convert to Torch's format
        local feat_idx = capt_idxs[1] + 1
        local x = feat_dset:partial({feat_idx, feat_idx},{1, feat_len})
        x = x:type(dtype):reshape(1, feat_len)

	-- Predict
  	local pred = torch.totable(model:sample({length=capt_len,h0=x}):reshape(1, capt_len))[1]

        -- Load the ground truth captions
        local truth = {}
        for _,capt_idx in pairs(capt_idxs) do
            -- Convert to 1-indexing
            capt_1idx = capt_idx + 1

            -- Load the caption
            truth[#truth + 1] = torch.totable(captions_dset:partial({capt_1idx,capt_1idx},{1,capt_len}))[1]
        end

        -- Accumulate the BLEU score
        score = score + bleu.idxBleu(pred, truth)
    end

    return score / num_imgs
end

function bleu.idxBleu(pred, truth)
--Internal function to compute the BLEU score given two tables of tokens. pred
--contains the predicted tokens, truth is a table, each element of which is a ground truth caption
    
    -- Strip both pred and truth of fluff tokens
    pred = bleu.stripCapt(pred)
    for idx, capt in pairs(truth) do
	truth[idx] = bleu.stripCapt(capt)
    end

    -- Count the max occurence of each n-gram in any ground truth caption
    local truthGrams = {}
    for _,capt in pairs(truth) do
	-- Count the n-grams for this caption
	local captGrams = {}
        for _,tok in pairs(capt) do	
	   if captGrams[tok] == nil then
	       captGrams[tok] = 1
	   else
	       captGrams[tok] = captGrams[tok] + 1
	   end
	end

	-- Compute the maximum n-grams counts across all captions
	for tok, count in pairs(captGrams) do
	    if truthGrams[tok] == nil then
		truthGrams[tok] = count
	    else
	    	truthGrams[tok] = math.max(truthGrams[tok], count)
	    end
	end
    end

    -- Count the matched n-grams from the predicted caption
    local matchedGrams = {}
    for _,tok in pairs(pred) do
        if truthGrams[tok] ~= nil then
            if matchedGrams[tok] == nil then
                matchedGrams[tok] = 1
            else
                matchedGrams[tok] = matchedGrams[tok] + 1
            end
        end
    end

    -- Tally the bleu score by restricting the matches to truthGrams
    local bleuMatched = 0
    for tok, count in pairs(matchedGrams) do
	bleuMatched = bleuMatched + math.min(count, truthGrams[tok])
    end

    return bleuMatched / #pred
end

function bleu.stripCapt(capt)
-- Internal function to strip a table of fluff tokens. Warning: this assumes any token less than or equal to fluffTok is fluff.

    -- HARDCODED tokens less than or equal to this are fluff
    local fluffTok = 3

    -- Rebuild the caption out-of-place
    local stripped = {}
    for _,tok in pairs(capt) do
	if tok > fluffTok then
	    stripped[#stripped + 1] = tok
	end
    end

    return stripped
end
