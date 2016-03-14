require 'torch'
require 'LanguageModel'
require 'AttentionCaptioningModel'
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

function bleu.getScore(checkpoint, h5name, split, mode, device, batchSize)
-- Return the bleu score over the specified split.
-- 
-- Parameters:
--   checkpoint: The torch checkpoint
--   h5name: The name of the .h5 file containing the data
--   split: Either bleu.split.train or bleu.split.val
--   mode: Either bleu.cpu, bleu.cuda, or bleu.cl
--   device: The device number to use, e.g. 1 for GPU 1.
--   batchSize: The number of images to caption at once.
--
-- Returns a number giving the average bleu score per caption.

    -- Print options
    print('Using split '..split..'...')

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
    print(string.format('Detected %d images...', num_imgs))

    -- Get the image size for attention features
    local image_size = nil
    if feat_dset:dataspaceSize()[3] ~= nil then
      image_size = {}
      for idx, size in pairs(feat_dset:dataspaceSize()) do
        if idx > 1 then
          image_size[#image_size + 1] = size
        end
        if idx > 2 then
          feat_len = feat_len * size
        end
      end
   end

    -- Initialize the model
    model:evaluate()
    model:resetStates()

    -- Process each image
    local preds = {}
    for i=1,num_imgs do 
        -- Print our progress
        if i % 5000 == 0 then
            print(string.format('Captioning image %d', i))
        end

        -- Predict in batch mode
        if preds[i] == nil then
            -- Get the size of this batch
            local thisBatchSize = math.min(batchSize, num_imgs - i + 1)

            -- Load each image's features
	    local x = nil
	    if image_size == nil then
	    	x = torch.zeros(thisBatchSize, feat_len):type(dtype)
	    else
	    	x = torch.zeros(thisBatchSize, image_size[1], image_size[2], image_size[3]):type(dtype)
	    end
            for j=1,thisBatchSize do
                -- Load the caption indices for this image
		local im_idx = i + j - 1
                local capt_idxs = torch.totable(map_dset:partial({im_idx, im_idx}, {1, idxs_per_img}))[1]
                -- Load the features for this image and convert to Torch's format
                local feat_idx = capt_idxs[1] + 1
                if feat_idx < 1 then
		    -- If we hit an error, continue on with the whole dataset up to this point
                    print(string.format('bleu.lua: Invalid map index detected at image %d. Ignoring the rest of the dataset', i))
		    num_imgs = i - 1
		    break
                end
		local feat = nil
		if image_size == nil then
                    feat = feat_dset:partial({feat_idx, feat_idx},{1, feat_len})
                    x[{j,{}}] = feat:type(dtype):reshape(1, feat_len)
		else
    		    feat = feat_dset:partial({feat_idx,feat_idx},{1,image_size[1]},
					  {1,image_size[2]},{1,image_size[3]})
                    x[{j,{},{},{}}] = feat:type(dtype):reshape(1, image_size[1], image_size[2], image_size[3])
		end
            end

	    -- Predict this batch
            local batchPred = torch.totable(model:sample({length=capt_len,h0=x}):reshape(thisBatchSize, capt_len))

	    -- Store the predictions
	    for j=1,thisBatchSize do
		local im_idx = i + j - 1
		assert(preds[im_idx] == nil)
		assert(batchPred[j] ~= nil)
		preds[im_idx] = batchPred[j]
	    end
        end
    end

    -- Compute the BLEU for each prediction
    local score = 0
    for i=1,num_imgs do
        -- Load the caption indices for this image
        local capt_idxs = torch.totable(map_dset:partial({i, i}, {1, idxs_per_img}))[1]

        -- Load the ground truth captions
        local truth = {}
        for _,capt_idx in pairs(capt_idxs) do
            -- Check for the 'no caption' flag
            if capt_idx < 0 then
                break
            end

            -- Convert to 1-indexing
            local capt_1idx = capt_idx + 1

            -- Load the caption
            capt = torch.totable(captions_dset:partial({capt_1idx,capt_1idx},{1,capt_len}))[1]

	   -- Add one to each token
	   for tok_idx,tok in pairs(capt) do
		capt[tok_idx] = tok + 1
	   end
	   
	   -- Store the caption
	   truth[#truth + 1] = capt
        end

        print('-----------PRED-----------')
        bleu.printCapt(preds[i], model)
        print('-----------TRUTH-----------')
        for _,capt in pairs(truth) do
	    bleu.printCapt(capt, model)
        end
    

        -- Accumulate the BLEU score
	assert(preds[i] ~= nil)
	thisScore, predMatched = bleu.idxBleu(preds[i], truth)
	bleu.printCapt(predMatched, model)
        score = score + thisScore
    end

    return score / num_imgs
end

function bleu.idxBleu(pred, truth)
--Internal function to compute the BLEU score given two tables of tokens. pred
--contains the predicted tokens, truth is a table, each element of which is a ground truth caption
--
-- Returns:
--   score: The BLEU score
--   predMatched: The matched tokens, ignoring repetitions

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

    -- Check for empty inputs
    numPred = 0
    numTruth = 0
    for _ in pairs(pred) do numPred = numPred + 1 end
    for _ in pairs(truthGrams) do numTruth = numTruth + 1 end
    if numTruth == 0 then
        return 1
    end
    if numPred == 0 then
        return 0
    end

    -- Count the matched n-grams from the predicted caption
    local predMatched = {}
    local matchedGrams = {}
    for _,tok in pairs(pred) do
        if truthGrams[tok] ~= nil then
	    predMatched[#predMatched + 1] = tok
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
    print('------------MATCHED--------------')
    print(string.format('%d / %d', bleuMatched, numPred))

    return bleuMatched / numPred, predMatched
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

function bleu.printCapt(capt, model)
-- Print a table of numbers as a caption
if next(capt) == nil then
	print('')
else
	print(model:decode_string(torch.Tensor(bleu.stripCapt(capt))))
end

end
