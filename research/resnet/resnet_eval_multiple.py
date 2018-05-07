# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
from __future__ import print_function, division
import time
import json

import cifar_input
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_string('log_root1', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('log_root2', '', 'Directory for checkpoints of m2.')
tf.app.flags.DEFINE_string('m1name', 'm1', 'Model1 Name')
tf.app.flags.DEFINE_string('m2name', 'm2', 'Model2 Name')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('n_trials', 200, 'Number of trials for performance testing.')

def evaluate(n_classes):
  """Eval loop."""
  batch_size = 128
  timed_results = {'m1_latency' : [], 'm2_latency' : [], 'batch_size' : batch_size}
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, batch_size, 'eval')
  hps = resnet_model.HParams(batch_size=batch_size // 2,
                       num_classes=n_classes,
                       min_lrn_rate=0.0001,
                       lrn_rate=0.1,
                       num_residual_units=5,
                       use_bottleneck=False,
                       weight_decay_rate=0.0002,
                       relu_leakiness=0.1,
                       optimizer='mom')
  sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess1)
  images1, images2 = tf.split(images, 2)
  labels1, labels2 = tf.split(labels, 2)
  images1 = images1.eval()
  print(images1)
  model1 = resnet_model.ResNet(hps, images1, labels1, 'eval')
  model2 = resnet_model.ResNet(hps, images2, labels2, 'eval')
  with tf.variable_scope(FLAGS.m1name) as scope:
    model1.build_graph()
  with tf.variable_scope(FLAGS.m2name) as scope:
    model2.build_graph()
  saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(FLAGS.m1name)))
  saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(FLAGS.m2name)))
  sess1.run(tf.initialize_all_variables())
  try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root1)
  except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
  tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  try:
      ckpt_state2 = tf.train.get_checkpoint_state(FLAGS.log_root2)
  except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
  tf.logging.info('Loading checkpoint %s', ckpt_state2.model_checkpoint_path)
  saver1.restore(sess1, ckpt_state.model_checkpoint_path)
  saver2.restore(sess1, ckpt_state2.model_checkpoint_path)
  for i in range(FLAGS.n_trials):
    start = time.time()
    predictions = sess1.run([model1.cost, model1.predictions, model1.labels])
    timed_results['m1_latency'].append(time.time() - start)
    start = time.time()
    predictions = sess1.run([model2.cost, model2.predictions, model2.labels])
    timed_results['m2_latency'].append(time.time() - start)
    print(str(i), str(timed_results['m1_latency'][-1]), str(timed_results['m2_latency'][-1]))
  with open('combined_latency.json', 'w') as out_f:
    json.dump(timed_results, out_f)

def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  with tf.device(dev):
    evaluate(num_classes)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
