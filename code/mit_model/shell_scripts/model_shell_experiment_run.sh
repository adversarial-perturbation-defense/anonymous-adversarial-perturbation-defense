#!/bin/bash
# The following code assumes that the working directory contains saved_model
#directory.
path=$(pwd)
model_save_dir="$path/saved_models/" #adv_trained_mit_model"
model_load_path="$path/saved_models/adv_trained_mit_model"
output_path="$path/outputs/output_experiments_2.txt"
echo "" > $output_path
stdbuf -oL -eL python experiment.py \
  --model_name=adv_trained_mit_model \
  --model_save_directory=$model_save_dir \
  --sample_test_point=True \
  --test_image_index=7503 \
  --starting_model_save_path=$model_load_path \
  --num_samples=10 \
  2> >(tee -a $output_path >&2)
# glog always output to stderr
