#!/bin/bash

config='MPT_MAE256'
script='ostrack'
python tracking/train.py --script ${script} --prompt 1 --config ${config} --save_dir YOUR_SAVE_PATH --mode multiple --nproc_per_node 2

dataset='lasot_extension_subset'
epoch='60'
python tracking/test.py ${script} ${config} --dataset ${dataset} --threads 2 --num_gpus 2 --epoch ${epoch}
