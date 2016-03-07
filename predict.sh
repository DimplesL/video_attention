th predict.lua -images test_images.txt \
               -checkpoint /data/checkpoints/checkpoint_250.t7 \
               -input_json /data/coco/coco_vocab.json \
               -sample_length 49 \
               -gpu 1
