#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os
import numpy as np
import h5py
import codecs
import string
import glob
import os
import math
import re
from collections import defaultdict
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--train_json', default='/data/coco/annotations/captions_train2014.json')
parser.add_argument('--val_json', default='/data/coco/annotations/captions_val2014.json')
parser.add_argument('--test_dir', default='/data/coco/test2014')
parser.add_argument('--train_dir', default='/data/coco/train2014')
parser.add_argument('--val_dir', default='/data/coco/val2014')
parser.add_argument('--output_h5', default='/data/coco/coco2.h5')
parser.add_argument('--output_json', default='/data/coco/coco_vocab2.json')
parser.add_argument('--subsample_percent', default=1.0)
parser.add_argument('--vocab_size', default=2048)
args = parser.parse_args()

h5 = h5py.File(args.output_h5, "w")

# Make the test dataset
test_files = glob.glob(os.path.join(args.test_dir,"*_f1.npy"))
test_files = np.random.choice(test_files, \
	size=math.floor(float(args.subsample_percent)*len(test_files)), \
	replace=False)
num_test = len(test_files)
test_names = h5.create_dataset('test_names', shape=(num_test,), dtype="S10")

# Populate the test dataset
feat_len = None
test_dset = None
for i,fname in enumerate(test_files):
  feats = np.load(fname).mean(axis=(1,2)).squeeze()
  if feat_len is None:
	# Get the length of a feature from the first file
	feat_len = feats.shape[0]
	test_dset = h5.create_dataset('test_feats', shape=(num_test,feat_len))
	print "Detected feature length %d from file %s" % (feat_len, fname)
  assert(len(feats.shape) == 1)
  assert(feat_len == feats.shape[0])
  test_dset[i] = feats
  test_names[i] = fname
  if i % 1000 == 0:
	print "Adding feature %d to test dataset" % (i)

print "Loading captions json..."
trainf = open(args.train_json)
valf = open(args.val_json)
train_json = json.load(trainf)
val_json = json.load(valf)
train_cs = np.random.choice(train_json['annotations'],size=math.floor(float(args.subsample_percent)*len(train_json['annotations'])),replace=False)
val_cs = np.random.choice(val_json['annotations'],size=math.floor(float(args.subsample_percent)*len(val_json['annotations'])),replace=False)
# subsample json files

captions = {'train':defaultdict(list), 'val':defaultdict(list)}
token_to_idx = {}
max_caption_length= -float('inf')
num_capts=defaultdict(int)
# Add start and null to the vocab
token_to_idx["<NULL>"] = len(token_to_idx)
token_to_idx["<START>"] = len(token_to_idx)
token_to_idx["<END>"] = len(token_to_idx)
token_to_idx["<UNK>"] = len(token_to_idx)
split_re = re.compile('\W+')
hist = defaultdict(int)

def isblank(s):
  return not (s and s.strip())

def tokenize_caption(caption):
  caption = caption.encode('ascii','ignore').strip().lower()
  caption_words= split_re.split(caption)
  caption_words = [w for w in caption_words if not isblank(w)]
  for word in caption_words:
    hist[word] += 1
    if word not in token_to_idx:
      token_to_idx[word] = len(token_to_idx)
  return caption_words

for name,js in [('train',train_cs),('val',val_cs)]:
  for ann in js:
    im_id = ann['image_id']
    caption = ann['caption']
    toks = tokenize_caption(caption)
    max_caption_length = max(max_caption_length, len(toks))
    captions[name][im_id].append(toks)
    num_capts[name] +=1

print "Tokenizing captions.."
# find most commonly used words in histogram
most_common = sorted(hist.iteritems(),key=lambda x: -x[1])
# limit to vocab size
vocab_hist = most_common[:args.vocab_size]
# calculate number known words vs total
num_known_words = sum([x[1] for x in vocab_hist])
num_total_words = sum([y for x,y in hist.iteritems()])
# create a set of known words

known_words = ["<NULL>","<START>","<END>","<UNK>"]
# add words from known set
known_words += [x[0] for x in vocab_hist]
# create token to index from known words
token_to_idx = dict([(w,i) for i,w in enumerate(known_words)])
# change to a default dict that outputs unkonwn if the word isnt in vocab
unk_idx = token_to_idx["<UNK>"]
# add start and end tokens to every caption
max_caption_length +=2 
idx_captions = {'train':defaultdict(list), 'val':defaultdict(list)}
for name, data in captions.iteritems():
  for imid, im_capts in data.iteritems():
    for capt in im_capts:
      capt = ["<START>"] + capt + ["<END>"]
      capt += ["<NULL>"]*(max_caption_length-len(capt))
      idx_captions[name][imid].append([token_to_idx[w] if w in token_to_idx else unk_idx for w in capt])
      

print "Using %d%% of captions." % math.floor(float(args.subsample_percent)*100)
print "Loaded %d training captions" % num_capts['train']
print "Loaded %d validation captions" % num_capts['val']
print "Max caption length: %d" % max_caption_length
print "Num different words: %d" % len(hist)
print "Vocab size of %d covers %f%% of words" % (args.vocab_size, 100*num_known_words/float(num_total_words))

# Save the vocabulary to a json file
print "Saving json"
idx_to_token = dict([(idx,token) for token,idx in token_to_idx.items()])
json_data = { "idx_to_token":idx_to_token,"token_to_idx":token_to_idx}

with open(args.output_json, 'w') as f:
  json.dump(json_data,f)

# Find the maximum number of captions for each image, for each dataset
im_max_capts = {'train': -1, 'val' : -1}
for dset_name, dset_capts in captions.iteritems():
    for im_id, im_captions in dset_capts.iteritems():
	im_max_capts[dset_name] = max(im_max_capts[dset_name], len(im_captions))

for dset_name, max_capts in im_max_capts.iteritems():
    print "Dataset %s has at most %d captions per image" % (dset_name, max_capts)

# Initialize maps from images to h5 caption/feature indices. -1 means no caption
num_train = num_capts['train']
num_val = num_capts['val']
maps = {'train' : -np.ones(shape=(num_train, im_max_capts['train']), dtype=np.uint64), 
        'val' : -np.ones(shape=(num_val, im_max_capts['val']), dtype=np.uint64)}
print "Writing hd5"
# create the datasets
# dataset will be N x max_caption_length
num_train = num_capts['train']
num_val = num_capts['val']
train_feats = h5.create_dataset('train_feats', shape=(num_train,feat_len))
val_feats = h5.create_dataset('val_feats', shape=(num_val,feat_len))
train_names = h5.create_dataset('train_names', shape=(num_train,), dtype="S10")
val_names = h5.create_dataset('val_names', shape=(num_val,), dtype="S10")
train_captions = h5.create_dataset('train_captions', shape=(num_train,max_caption_length),dtype='i4')
val_captions = h5.create_dataset('val_captions', shape=(num_val,max_caption_length),dtype='i4')

# Add each caption to the datasets and build the maps
for dset_name,dset_capts in idx_captions.iteritems():
  curr_capt = 0
  map_idx = defaultdict(int);
  for im_id, im_captions in dset_capts.iteritems():
    # If necessary, add this image to the map_idx lookup table.
    if im_id not in map_idx:
	map_idx[im_id] = len(map_idx)

    # Get the name of the .npy features file
    if dset_name =="train":
      fname = "COCO_%s2014_%012d_f1.npy" % (dset_name, int(im_id))
      fname = os.path.join(args.train_dir,fname)
    elif dset_name == "val":
      fname = "COCO_%s2014_%012d_resnet50.npy" % (dset_name, int(im_id))
      fname = os.path.join(args.val_dir,fname)

    # Try to load the file
    try:
      feats = np.load(fname).mean(axis=(1,2))
    except:
      print "ERROR, unable to load features: %s, shape: %s" % (fname,np.load(fname).shape)
      exit(1)
    assert(len(feats.shape) == 1)
    assert(feat_len == feats.shape[0])

    # Add each caption to the map/dataset
    for caption in im_captions:
      # Add the token to the map
      maps[dset_name][map_idx[im_id], im_captions.index(caption)] = curr_capt

      # Add the caption to the dataset
      if dset_name == 'train':
        train_feats[curr_capt] = feats
	train_names[curr_capt] = fname
        train_captions[curr_capt] = np.array(caption,dtype=np.uint32)
      if dset_name == 'val':
        val_feats[curr_capt] = feats
	val_names[curr_capt] = fname
        val_captions[curr_capt] = np.array(caption,dtype=np.uint32)
      curr_capt += 1
      if curr_capt % 1000 == 0:
        print "Saved %d %s captions out of %d" % (curr_capt,dset_name,num_capts[dset_name])

# Add the maps to the h5 file
for dset_name, dset_map in maps.iteritems():
	h5.create_dataset(dset_name + '_map', data=dset_map)

