th predict.lua -images bigger_ims \
               -checkpoint /data/checkpoints/64_1536_2_2500_10000.t7\
               -input_json /data/coco/coco_vocab.json \
               -sample_length 51 \
               -gpu 3
