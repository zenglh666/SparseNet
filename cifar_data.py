from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cifar10_data_dir', 'E:/data2/cifar/cifar10/cifar-10-batches-bin',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('cifar100_data_dir', 'E:/data2/cifar/cifar100/cifar-100-binary',
                           """Path to the imagenet data directory.""")

class Cifar10Data():
  """ImageNet data set."""

  def __init__(self, name, subset):
    assert subset in self.available_subsets(), self.available_subsets()
    self.name = name
    self.subset = subset
    self.size = 32
    self.resize_size = 32
    self.crop_size = 24
    self.label_bytes = 1
    self.image_bytes = self.size * self.size * 3

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 50000
    if self.subset == 'validation':
      return 10000

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if self.subset=='train':
      data_files = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    elif self.subset=='validation':
      data_files = [os.path.join(data_dir, 'test_batch.bin')]

    for f in data_files:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)  

    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = self.label_bytes + self.image_bytes
    return tf.FixedLengthRecordReader(record_bytes=record_bytes)

  def parse_from_string(self, example_serialized):
    record = tf.decode_raw(example_serialized, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.slice(record, [0], [self.label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
      tf.slice(record, [self.label_bytes], [self.image_bytes]),
      [3, self.size, self.size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(depth_major, [1, 2, 0])
    return image, label



class Cifar100Data():
  """ImageNet data set."""

  def __init__(self, name, subset):
    assert subset in self.available_subsets(), self.available_subsets()
    self.name = name
    self.subset = subset
    self.size = 32
    self.resize_size = 32
    self.crop_size = 24
    self.label_bytes = 1
    self.image_bytes = self.size * self.size * 3

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 100

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 50000
    if self.subset == 'validation':
      return 10000

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if self.subset=='train':
      data_files = [os.path.join(FLAGS.cifar100_data_dir, 'train.bin')]
    elif self.subset=='validation':
      data_files = [os.path.join(FLAGS.cifar100_data_dir, 'test.bin')]

    for f in data_files:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)  

    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = self.label_bytes + self.image_bytes + 1
    return tf.FixedLengthRecordReader(record_bytes=record_bytes)

  def parse_from_string(self, example_serialized):
    record = tf.decode_raw(example_serialized, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.slice(record, [1], [self.label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
      tf.slice(record, [self.label_bytes + 1], [self.image_bytes]),
      [3, self.size, self.size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(depth_major, [1, 2, 0])
    return image, label