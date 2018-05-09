Build code: bazel build -c opt --config=cuda research/resnet/...
Train model1: bazel-bin/research/resnet/resnet_main --train_data_path=cifar-10-batches-bin/data_batch* --log_root=/tmp/resnet_model1 --train_dir=/tmp/resnet_model1/train --dataset='cifar10' --num_gpus=1 --model_name=m1

Train model2: bazel-bin/research/resnet/resnet_main --train_data_path=cifar-10-batches-bin/data_batch* --log_root=/tmp/resnet_model2 --train_dir=/tmp/resnet_model2/train --dataset='cifar10' --num_gpus=1 --model_name=m2

Evaluating single model: bazel-bin/research/resnet/resnet_main --eval_data_path=cifar-10-batches-bin/test_batch.bin --log_root=/tmp/resnet_model1 --model_name=m1 --dataset='cifar10' --num_gpus=0 --n_trials=50 --mode=eval --batch_size=64

Evaluating combined preprocessing: bazel-bin/research/resnet/resnet_eval_multiple --eval_data_path=cifar-10-batches-bin/test_batch.bin --log_root1=/tmp/resnet_model1 --log_root2=/tmp/resnet_model2 --m1name=m1 --m2name=m2 --eval_dir=/tmp/resnet_model/test --dataset='cifar10' --num_gpus=0 --batch_size=128

Alternatively, run eval_all_batches.sh

