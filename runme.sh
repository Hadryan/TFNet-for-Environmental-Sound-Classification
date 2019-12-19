#!/bin/bash
# You need to modify this path to your downloaded dataset directory
# Download ESC-10 and ESC-50 : https://github.com/karolpiczak/ESC-50
# Download UrbanSound8k : https://urbansounddataset.weebly.com/urbansound8k.html
DATASET_DIR='/.../ESC-50'
#DATASET_DIR='/.../ESC-10'
#DATASET_DIR='/.../UrbanSound8k'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/workspace'

# Hyper-parameters
GPU_ID=0
MODEL_TYPE='TFNet'
BATCH_SIZE=32

############ Train and test on ESC50 dataset ############
# Calculate feature
python util_esc50/feature.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train and test
CUDA_VISIBLE_DEVICES=$GPU_ID python util/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda
