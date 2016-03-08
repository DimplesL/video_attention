th train.lua -input_h5 /data/coco/coco.h5 \
             -input_json /data/coco/coco_vocab.json \
             -gpu 2 \
             -batch_size 32 \
             -seq_length 51 \
             -model_type "lstm" \
             -wordvec_size 1536 \
             -rnn_size 2048 \
             -num_layers 1 \
             -learning_rate 0.001 \
             -checkpoint_every 250 \
             -lr_decay_every 5 \
             -batchnorm 0\
             -max_epochs 500 
