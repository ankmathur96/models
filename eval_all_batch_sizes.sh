#!/bin/bash
bazel build -c opt --config=cuda research/resnet/...
for i in 2 4 8 16 32 64 128 256 512 1024
do
  bazel-bin/research/resnet/resnet_eval_multiple --eval_data_path=cifar-10-batches-bin/test_batch.bin --log_root1=/tmp/resnet_model1 --log_root2=/tmp/resnet_model2 --m1name=m1 --m2name=m2 --eval_dir=/tmp/resnet_model/test --dataset='cifar10' --num_gpus=0 --n_trials=50 --batch_size=$i
done 
