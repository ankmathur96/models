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
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root1', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('log_root2', '', 'Directory for checkpoints of m2.')
tf.app.flags.DEFINE_string('m1name', 'm1', 'Model1 Name')
tf.app.flags.DEFINE_string('m2name', 'm2', 'Model2 Name')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')


def train(n_classes):
  """Training loop."""
  hps = resnet_model.HParams(batch_size=128, num_classes=n_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def evaluate(n_classes):
  """Eval loop."""
  batch_size = 128
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, batch_size, FLAGS.mode)
  print(images)
  hps = resnet_model.HParams(batch_size=batch_size/2,
                       num_classes=n_classes,
                       min_lrn_rate=0.0001,
                       lrn_rate=0.1,
                       num_residual_units=5,
                       use_bottleneck=False,
                       weight_decay_rate=0.0002,
                       relu_leakiness=0.1,
                       optimizer='mom')
  images1, images2 = tf.split(images, 2)
  labels1, labels2 = tf.split(labels, 2)
  model1 = resnet_model.ResNet(hps, images1, labels1, FLAGS.mode)
  model2 = resnet_model.ResNet(hps, images2, labels2, FLAGS.mode)
  sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with tf.variable_scope(FLAGS.m1name) as scope:
    model1.build_graph()
  saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(FLAGS.m1name)))

  tf.train.start_queue_runners(sess1)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root1)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root1)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    var_list = checkpoint_utils.list_variables(FLAGS.log_root1)
    print('ROOT1:', var_list)
    try:
      ckpt_state2 = tf.train.get_checkpoint_state(FLAGS.log_root2)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state2 and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root2)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state2.model_checkpoint_path)
    var_list = checkpoint_utils.list_variables(FLAGS.log_root2)
    print('ROOT2:', var_list)
    saver1.restore(sess1, ckpt_state.model_checkpoint_path)
    print('Restored Model 1')
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess2)
    with tf.variable_scope(FLAGS.m2name) as scope:
        model2.build_graph()
    saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(FLAGS.m2name)))
    saver2.restore(sess2, ckpt_state2.model_checkpoint_path)
    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      loss, predictions, truth = sess1.run([model1.cost, model1.predictions, model1.labels])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]
    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    tf.logging.info('loss1: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))

    for _ in six.moves.range(FLAGS.eval_batch_count):
      loss, predictions, truth = sess2.run([model2.cost, model2.predictions, model2.truth])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]


    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)


    tf.logging.info('loss2: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))

    if FLAGS.eval_once:
      break

    time.sleep(60)


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
    if FLAGS.mode == 'train':
      train(num_classes)
    elif FLAGS.mode == 'eval':
      evaluate(num_classes)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
