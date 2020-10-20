#!/usr/bin/env bash

data_dir="fb15K/head-10"
aggregate_type='attention'
iter_routing=0
use_relation=1
score_function='TransE'
n_neg=1
loss_function='margin'
margin=300.0
weight_decay=1e-3
corrupt_mode='both'
max_neighbor=64
embedding_dim=100
batch_size=1024
learning_rate=1e-3
num_epoch=500
epoch_per_checkpoint=1
gpu_device="0"
gpu_fraction="0.2"
hparam="hparam"
save_dir="./checkpoints/fb15K/head-10/attention/report/"
run_log="data/fb15K/head-10/run.log"
