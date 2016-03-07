#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os
import numpy as np
import h5py
import codecs
import string
import glob
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--train_json', default='/data/coco/annotations/captions_train2014.json')
parser.add_argument('--val_json', default='/data/coco/annotations/captions_val2014.json')
parser.add_argument('--test_dir', default='/data/coco/test2014')
parser.add_argument('--train_dir', default='/data/coco/train2014')
parser.add_argument('--val_dir', default='/data/coco/val2014')
parser.add_argument('--output_h5', default='/data/coco/coco2.h5')
parser.add_argument('--output_json', default='/data/coco/coco_vocab2.json')
args = parser.parse_args()

h5 = h5py.File(args.output_h5, "w")

# Make the test dataset
test_files = glob.glob(os.path.join(args.test_dir,"*_f1.npy"))
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

# Read the input captions
trainf = open(args.train_json)
valf = open(args.val_json)
train_json = json.load(trainf)
val_json = json.load(valf)

captions = {'train':defaultdict(list), 'val':defaultdict(list)}
token_to_idx = {}
max_caption_length= -float('inf')
num_capts=defaultdict(int)

# Add the inputs captions to dictionaries
for name,js in [('train',train_json),('val',val_json)]:
  for ann in js['annotations']:
    im_id = ann['image_id']
    caption = ann['caption']
    caption = caption.encode('ascii','ignore').strip().lower()
    caption = caption.translate(None,string.punctuation)
    for word in caption.split():
      if word not in token_to_idx:
        token_to_idx[word] = len(token_to_idx)
    max_caption_length = max(max_caption_length, len(caption))
    captions[name][im_id].append([token_to_idx[word] for word in caption.split()])
    num_capts[name] +=1

print "Loaded %d training captions" % num_capts['train']
print "Loaded %d validation captions" % num_capts['val']
# Add start and null to the vocab
start_tok = "<START>"
end_tok = "<NULL>"
token_to_idx[start_tok] = len(token_to_idx)
token_to_idx[end_tok] = len(token_to_idx)

# Ensure our vocabulary will fit in an int
if len(token_to_idx) > 2**32:
  print "You fucked up, son"
  exit(1)

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

# create the training and validation datasets
# dataset will be N x max_caption_length
train_feats = h5.create_dataset('train_feats', shape=(num_train,feat_len))
train_names = h5.create_dataset('train_names', shape=(num_train,), dtype="S10")
val_feats = h5.create_dataset('val_feats', shape=(num_val,feat_len))
val_names = h5.create_dataset('val_names', shape=(num_val,), dtype="S10")
train_captions = h5.create_dataset('train_captions', shape=(num_train,max_caption_length),dtype='i4')
val_captions = h5.create_dataset('val_captions', shape=(num_val,max_caption_length),dtype='i4')

# Add each caption to the datasets and build the maps
map_idx = defaultdict(int);
for dset_name,dset_capts in captions.iteritems():
  curr_capt = 0
  for im_id, im_captions in dset_capts.iteritems():
    # If necessary, add this image to the map_idx lookup table
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
      feats = np.load(fname).mean(axis=(1,2)).squeeze()
    except:
      print "FUCK"
      print fname
      print np.load(fname).shape
      exit(1)
    assert(len(feats.shape) == 1)
    assert(feat_len == feats.shape[0])

    # Add each caption to the map/dataset
    for caption in im_captions:
      # Add the token to the map
      maps[dset_name][map_idx[im_id], im_captions.index(caption)] = curr_capt

      # Pad the caption will null tokens
      caption += [token_to_idx[end_tok]]*(max_caption_length-len(caption))

      # Add the caption to the dataset
      if dset_name == 'train':
        train_feats[curr_capt] = feats
	train_names[curr_capt] = fname
        train_captions[curr_capt] = np.array(caption,dtype=np.uint32)
      elif dset_name == 'val':
        val_feats[curr_capt] = feats
	val_names[curr_capt] = fname
        val_captions[curr_capt] = np.array(caption,dtype=np.uint32)
      if curr_capt % 1000 == 0:
	print "Adding caption %d to dataset %s" % (curr_capt, dset_name)
      curr_capt += 1

# Add the maps to the h5 file
for dset_name, dset_map in maps.iteritems():
	h5.create_dataset(dset_name + '_map', data=dset_map)

# Make the output JSON vocabulary
idx_to_token = [(idx,token) for token,idx in token_to_idx.items()]
json_data = { "idx_to_token":idx_to_token,"token_to_idx":token_to_idx}

with open(args.output_json, 'w') as f:
  json.dump(json_data,f)
