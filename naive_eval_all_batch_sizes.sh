#!/bin/bash
bazel build -c opt --config=cuda research/resnet/...
for i in 2 4 8 16 32 64 128 256 512 1024
do
  bazel-bin/research/resnet/resnet_main --eval_data_path=cifar-10-batches-bin/test_batch.bin --log_root=/tmp/resnet_model1 --model_name=m1 --dataset='cifar10' --num_gpus=0 --n_trials=50 --mode=eval --batch_size=$i
done
