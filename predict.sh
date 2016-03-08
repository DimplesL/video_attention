th predict.lua -images train_images.txt \
               -checkpoint /data/checkpoints/checkpoint_500.t7 \
               -input_json /data/coco/tiny_coco_vocab.json \
               -sample_length 16 \
               -gpu 1
