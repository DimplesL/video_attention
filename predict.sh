th predict.lua -images train_images.txt \
               -checkpoint /data/checkpoints/checkpoint_2500.t7 \
               -input_json /data/coco/small_coco_vocab.json \
               -sample_length 45 \
               -gpu 2
