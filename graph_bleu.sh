#!/bin/bash

# BLEU command
BLEU="th /home/blaine/video_attention/eval_bleu.lua"

# Run on various checkpooints
for iter in 1000 3000 5000 7000 10000
do
	# Get the checkpoint file
	CHK="/data/checkpoints/64_1536_2_2500_${iter}.t7"
	
	for split in "train" "val"
	do
		# Get the log file
		LOG="/home/ubuntu/log/blue${iter}${split}.txt"

		# Run BLEU
		${BLEU} -checkpoint ${CHK} -split ${split} | tee ${LOG}
	done
done
