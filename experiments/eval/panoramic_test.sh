#!/usr/bin/env bash

python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights ./output/exp_dir/pano_2018_train/VGGnet_fast_rcnn_iter_30000.ckpt \
--cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
--network VGGnet_test
