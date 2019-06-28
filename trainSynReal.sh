#!/bin/bash
DATASETS=${1}_train5k_test10k_lmdb
CHECK_DIR=/data/jing_liang/CRNN
SUFFIX=""
GPU=${2}
BATCH=128
PRETRAINED=-1
EPOCH=100
IMG_H=48
IMG_W=600



python train.py \
--trainRoot datasets/${DATASETS}/trainB \
--valRoot   datasets/${DATASETS}/testB \
--batchSize ${BATCH} \
--imgH ${IMG_H} \
--imgW ${IMG_W} \
--nepoch ${EPOCH} \
--expr_dir ${CHECK_DIR}/${DATASETS}${SUFFIX} \
--adadelta \
--random_sample \
--displayInterval 30 \
--save_freq 1 \
--val_freq 1 \
--cuda ${GPU} \
--pretrained ${PRETRAINED}

