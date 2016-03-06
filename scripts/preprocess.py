# -*- coding: utf-8 -*-

# MODIFIED from original file.
# Options:
#   --train_features - The parent directory of the training .npy features
#   --val_features - The parent directory of the validation .npy features
#   --test_features - The parent directory of the testing .npy features
#   --train_annotations - The .json file of the training annotations
#   --val_annotations - The .json file of the validation annotations
#   --output_h5 - The output .h5 file containing the formatted features
#   --output_json - The output .json file containing the vocabulary

import argparse, json, os
import numpy as np
import h5py
import codecs


parser = argparse.ArgumentParser()
#parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
#parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
#parser.add_argument('--output_json', default='data/tiny-shakespeare.json')
#parser.add_argument('--val_frac', type=float, default=0.1)
#parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default='utf-8')
#---New args----
parser.add_argument('--train_features', default = /data/coco/train2014)
parser.add_argument('--val_features', default = /data/coco/val2014)
parser.add_argument('--test_features', default = /data/coco/test2014)
parser.add_argument('--train_annotations', \
    default = /data/annotations/captions_train2014.json)
parser.add_argument('--val_annotations', \
    default = /data/annotations/captions_val2014.json)
parser.add_argument('--output_h5', default='../data/coco.h5')
parser.add_argument('--output_json', default='../data/coco.json')
#---------------
args = parser.parse_args()

if __name__ == '__main__':
  if args.encoding == 'bytes': args.encoding = None

  # First go the file once to see how big it is and to build the vocab
  token_to_idx = {}
  total_size = 0
  num_captions = {args.train_annotations : 0, args.val_annotations: 0}
  max_caption = 0
  annotations = [args.train_annotations, args.val_annotations]
  for json_name in annotations:
    # Open the .json file
    with codecs.open(json_name, 'r', args.encoding) as f:
        # Parse the contents
        parsed_json = json.load(f)

        # Get each caption character
        caption = "" #TODO get the caption somehow
        max_caption = max(max_caption, len(caption))
        total_size += len(caption)
        num_captions[json_name] += 1
        for char in caption:
            if char not in token_to_idx:
                token_to_idx[char] = len(token_to_idx) + 1
        
  #total_size = 0
  #with codecs.open(args.input_txt, 'r', args.encoding) as f:
  #  for line in f:
  #    total_size += len(line)
  #    for char in line:
  #      if char not in token_to_idx:
  #        token_to_idx[char] = len(token_to_idx) + 1

  # Now we can figure out the split sizes
  #val_size = int(args.val_frac * total_size)
  #test_size = int(args.test_frac * total_size)
  #train_size = total_size - val_size - test_size
 
  if not args.quiet:
    print 'Total vocabulary size: %d' % len(token_to_idx)
    print 'Total tokens in file: %d' % total_size
    #print '  Training size: %d' % train_size
    #print '  Val size: %d' % val_size
    #print '  Test size: %d' % test_size

  # Choose the datatype based on the vocabulary size
  dtype = np.uint8
  if len(token_to_idx) > 255:
    dtype = np.uint32
  if not args.quiet:
    print 'Using dtype ', dtype

  # Just load data into memory ... we'll have to do something more clever
  # for huge datasets but this should be fine for now
  train = np.zeros(train_size, dtype=dtype)
  val = np.zeros(val_size, dtype=dtype)
  test = np.zeros(test_size, dtype=dtype)
  splits = [train, val, test]

  # Go through the files again and write data to the HDF5 file
  feat_len = 0 # TODO read the length of a feature
  num_test = 0 # TODO get the length of the testing set, or resize text_x on the fly
  num_train = num_captions[args.train_annotations];
  num_val = num_captions[args.val_annotations];
  with h5py.File(args.output_h5, 'w') as f:
    # Initialize the datasets
    train_x = f.create_dataset('train_x', shape=(num_train, feat_len))
    val_x = f.create_dataset('val_x', shape=(num_val, feat_len))
    test_x = f.create_dataset('test_x', shape=(num_test, feat_len))
    train_y = f.create_dataset('train_y', shape=(num_train, max_caption), dtype = dtype)
    val_y = f.create_dataset('val', shape=(num_val, max_caption), dtype = dtype)

    # Add each feature and caption
    for json_name in annotations:
        with codecs.open(json_name, 'r', args.encoding) as j:
            parsed_json = json.load(j)
            caption = "" #TODO get the caption somehow
            # Update the y label

            # Get the x label according this file (JSON 'image_id')
                

  # Go through the file again and write data to numpy arrays
  #split_idx, cur_idx = 0, 0
  #with codecs.open(args.input_txt, 'r', args.encoding) as f:
  #  for line in f:
  #    for char in line:
  #      splits[split_idx][cur_idx] = token_to_idx[char]
  #      cur_idx += 1
  #      if cur_idx == splits[split_idx].size:
  #        split_idx += 1
  #        cur_idx = 0

  # Write data to HDF5 file
  #with h5py.File(args.output_h5, 'w') as f:
  #  f.create_dataset('train', data=train)
  #  f.create_dataset('val', data=val)
  #  f.create_dataset('test', data=test)

  # For 'bytes' encoding, replace non-ascii characters so the json dump
  # doesn't crash
  if args.encoding is None:
    new_token_to_idx = {}
    for token, idx in token_to_idx.iteritems():
      if ord(token) > 127:
        new_token_to_idx['[%d]' % ord(token)] = idx
      else:
        new_token_to_idx[token] = idx
    token_to_idx = new_token_to_idx

  # Dump a JSON file for the vocab
  json_data = {
    'token_to_idx': token_to_idx,
    'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
  }
  with open(args.output_json, 'w') as f:
    json.dump(json_data, f)
