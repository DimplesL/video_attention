#!/bin/bash

# BLEU command
BLEU="th /home/blaine/video_attention/eval_bleu.lua"

# H5 file
H5="/data/coco/coco_bgr_noavg.h5"

# Run on various checkpoints
for iter in 50
do
	# Get the checkpoint file
	CHK="/data/checkpoints/64_1536_1_2500_att_${iter}.t7"
	
	for split in "train" "val"
	do
		# Get the log file
		LOG="/home/ubuntu/log/bleu${iter}${split}att.txt"

		# Run BLEU
		${BLEU} -h5 ${H5} -checkpoint ${CHK} -split ${split} | tee ${LOG}
	done
done
