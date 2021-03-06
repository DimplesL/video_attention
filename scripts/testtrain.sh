th train.lua -input_h5 /data/coco/coco.h5 \
             -input_json /data/coco/coco_vocab.json \
             -gpu 1 \
             -batch_size 64 \
             -seq_length 51 \
             -model_type "lstm" \
             -wordvec_size 1024 \
             -rnn_size 2048 \
             -num_layers 1 \
             -learning_rate 0.001 \
             -checkpoint_every 250 \
             -lr_decay_every 1000 \
             -batchnorm 0\
             -max_epochs 500 
