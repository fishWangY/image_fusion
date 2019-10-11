# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import fusion_utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def fusion(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=tf.AUTO_REUSE,
                 create_aux_logits=True,
                 scope='InceptionV3',
                 global_pool=False):
    """
    The default image size used to train this network is 299x299.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer
            is omitted and the input features to the logits layer (before dropout)
            are returned instead.
        is_training: whether is training or not.
        dropout_keep_prob: the percentage of activation values that are retained.
        min_depth: Minimum depth value (number of channels) for all convolution ops.
            Enforced when depth_multiplier < 1, and not an active constraint when
            depth_multiplier >= 1.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops. The value must be greater than zero. Typical
            usage will be to set this value in (0, 1) to reduce the number of
            parameters or computation cost of the model.
        prediction_fn: a function to get predictions out of logits.
        spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
            shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
        create_aux_logits: Whether to create the auxiliary logits.
        scope: Optional variable_scope.
        global_pool: Optional boolean flag to control the avgpooling before the
            logits layer. If false or unset, pooling is done with a fixed window
            that reduces default-sized inputs to 1x1, while larger inputs lead to
            larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
        net: a Tensor with the logits (pre-softmax activations) if num_classes
            is a non-zero integer, or the non-dropped-out input to the logits layer
            if num_classes is 0 or None.
        end_points: a dictionary from components of the network to the corresponding
            activation.

    Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
    """
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

            # stage 1
            # high frequency process
            net_h = slim.conv2d(inputs, 4, [5, 5], padding='SAME', scope='conv_H1',
                                weights_initializer=initializers.gaussian(uniform=True),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_h = slim.conv2d(net_h, 4, [5, 5], padding='SAME', scope='conv_H2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_h = slim.conv2d(net_h, 1, [5, 5], padding='SAME', scope='conv_H3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # lower frequency process
            net_l = slim.conv2d(inputs, 4, [5, 5], padding='SAME', scope='conv_L1',
                                weights_initializer=initializers.gaussian(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 4, [5, 5], padding='SAME', scope='conv_L2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 1, [5, 5], padding='SAME', scope='conv_L3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # concat and output
            net_hl = tf.concat([net_h, net_l], axis=-1)
            net = slim.conv2d(net_hl, 1, [5, 5], padding='SAME', scope='conv_C1',
                              weights_initializer=initializers.variance_scaling_initializer(),
                              biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['fusion/conv_H3'] = net_h
            end_points['fusion/conv_L3'] = net_l

            # stage 2
            net_h = slim.conv2d(net, 4, [5, 5], padding='SAME', scope='conv_H1',
                                weights_initializer=initializers.gaussian(uniform=True),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh, reuse=False)
            net_h = slim.conv2d(net_h, 4, [5, 5], padding='SAME', scope='conv_H2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_h = slim.conv2d(net_h, 1, [5, 5], padding='SAME', scope='conv_H3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # lower frequency process
            net_l = slim.conv2d(net, 4, [5, 5], padding='SAME', scope='conv_L1',
                                weights_initializer=initializers.gaussian(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 4, [5, 5], padding='SAME', scope='conv_L2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 1, [5, 5], padding='SAME', scope='conv_L3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # concat and output
            net_hl = tf.concat([net_h, net_l], axis=-1)
            net = slim.conv2d(net_hl, 1, [5, 5], padding='SAME', scope='conv_C1',
                              weights_initializer=initializers.variance_scaling_initializer(),
                              biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # stage 3
            net_h = slim.conv2d(net, 4, [5, 5], padding='SAME', scope='conv_H1',
                                weights_initializer=initializers.gaussian(uniform=True),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh, reuse=False)
            net_h = slim.conv2d(net_h, 4, [5, 5], padding='SAME', scope='conv_H2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_h = slim.conv2d(net_h, 1, [5, 5], padding='SAME', scope='conv_H3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # lower frequency process
            net_l = slim.conv2d(net, 4, [5, 5], padding='SAME', scope='conv_L1',
                                weights_initializer=initializers.gaussian(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 4, [5, 5], padding='SAME', scope='conv_L2',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)
            net_l = slim.conv2d(net_l, 1, [5, 5], padding='SAME', scope='conv_L3',
                                weights_initializer=initializers.variance_scaling_initializer(),
                                biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

            # concat and output
            net_hl = tf.concat([net_h, net_l], axis=-1)
            net = slim.conv2d(net_hl, 1, [5, 5], padding='SAME', scope='conv_C1',
                              weights_initializer=initializers.variance_scaling_initializer(),
                              biases_initializer=init_ops.zeros_initializer(), activation_fn=nn.tanh)

    return net, end_points


fusion.default_image_size = 256


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


fusion_arg_scope = fusion_utils.fusion_arg_scope()
