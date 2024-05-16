#!/bin/bash

# modify the dataset argument if you want to try other datasets.
# dataset: ['pascal', 'cityscapes', 'coco', 'ade20k']
dataset=$1
config=configs/${dataset}.yaml
ckpt_path=$2

python evaluate.py --config $config --ckpt-path $ckpt_path