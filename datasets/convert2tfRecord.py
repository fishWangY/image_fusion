from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import argparse
import tensorflow as tf

from datasets import dataset_utils


def parser_auguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=None, help='Where train dataset saved.')
    parser.add_argument('--train_dir', type=str, default=None, help='Where training tfRecord data saved.')
    parser.add_argument('--val_dir', type=str, default=None, help='Where validation tfRecord data saved.')
    parser.add_argument('--num_shards', type=int, default=5, help='Nums of tfRecord files')

    return parser.parse_args(argv)


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'fusion_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir, num_shards):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True, visible_device_list='3')
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))

                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        example = dataset_utils.imageOnly_to_tfexample(
                            image_data, b'jpg', height, width)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def main(args):

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.val_dir):
        os.makedirs(args.val_dir)

    # where training dataset saved.
    train_dir = os.path.join(args.data_dir, 'train')
    # where testing dataset saved.
    val_dir = os.path.join(args.data_dir, 'val')

    training_filenames = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
    validation_filenames = [os.path.join(val_dir, x) for x in os.listdir(val_dir)]

    _convert_dataset('train', training_filenames, args.train_dir, args.num_shards)
    _convert_dataset('validation', validation_filenames, args.val_dir, args.num_shards)

    print('\nFinished converting dataset!')


if __name__ == '__main__':
    main(parser_auguments(sys.argv[1:]))