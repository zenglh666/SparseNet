# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Small library that points to the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('imagenet_train_data_dir', 'F:/data/imagenet_scale256_train_tfrecord',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('imagenet_valid_data_dir', 'F:/data/imagenet_scale256_valid_tfrecord',
                           """Path to the imagenet data directory.""")

class ImagenetData():
  """ImageNet data set."""

  def __init__(self, name, subset):
    assert subset in self.available_subsets(), self.available_subsets()
    self.name = name
    self.subset = subset
    self.resize_size = 256
    self.crop_size = 224

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 1000

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1281167
    if self.subset == 'validation':
      return 50000

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if self.subset=='train':
      tf_record_pattern = os.path.join(FLAGS.imagenet_train_data_dir, '%s-*' % self.subset)
    elif self.subset=='validation':
      tf_record_pattern = os.path.join(FLAGS.imagenet_valid_data_dir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s' % (self.name, self.subset))
      self.download_message()
      exit(-1)
      
    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    return tf.TFRecordReader()

  def __parse_example_proto(self, example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

    Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']

  def __parse_scale_example_proto(self, example_serialized):
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/label'], dtype=tf.int32)

    return features['image/encoded'], label

  def __decode_jpeg(self, image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope, default_name='decode_jpeg'):
      # Decode the string as an RGB JPEG.
      # Note that the resulting image contains an unknown height and width
      # that is set dynamically by decode_jpeg. In other words, the height
      # and width of image is unknown at compile-time.
      image = tf.image.decode_jpeg(image_buffer, channels=3)

      # After this point, all image pixels reside in [0,1)
      # until the very end, when they're rescaled to (-1, 1).  The various
      # adjust_* ops all require this range for dtype float.
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return image

  def parse_from_string(self, example_serialized):
    if self.name == 'imagenet_scale':
      image_buffer, label_index = self.__parse_scale_example_proto(example_serialized)
    elif self.name == 'imagenet':
      image_buffer, label_index, bbox, _ = self.__parse_example_proto(example_serialized)
      label_index = tf.subtract(label_index, 1)
    image = self.__decode_jpeg(image_buffer)
    return image, label_index

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any ImageNet %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the ImageNet data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take several hours depending on the speed of your '
          'computer and network connection\n')
    print('Please see README.md for instructions on how to build '
          'the ImageNet dataset using download_and_preprocess_imagenet.\n')
    print('Note that the raw data size is 300 GB and the processed data size '
          'is 150 GB. Please ensure you have at least 500GB disk space.')
