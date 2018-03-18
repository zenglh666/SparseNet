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
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_preprocess_threads', 16,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 8,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_string('mean_file', 'E:\\data2\\imagenet_mean.npy',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_boolean('distort_color',False,
                            '''If we distort color''')

def create_data_batch(dataset, batch_size, num_preprocess_threads=None):
    if dataset.subset == 'train':
      images, labels = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
    elif dataset.subset == 'validation':
      images, labels = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
    return images, labels

def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=None):
  """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].

  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')

    # Create filename_queue
    if train:
      filename_queue = tf.train.string_input_producer(data_files)
    else:
      filename_queue = tf.train.string_input_producer(data_files)
    if num_preprocess_threads is None:
      num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = FLAGS.num_readers

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 16 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 16 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = dataset.reader()
      _, example_serialized = reader.read(filename_queue)

    if dataset.name=='imagenet':
      mean_array = np.transpose(tf.divide(np.load(FLAGS.mean_file), 256.), (2,1,0))
      mean_tensor = tf.convert_to_tensor(mean_array, tf.float32)
    else:
      mean_tensor = None

    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      image, label_index = dataset.parse_from_string(example_serialized)
      image = image_preprocessing(
        image, train, dataset.resize_size, dataset.crop_size, mean_tensor, thread_id)
      images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity= num_preprocess_threads * batch_size)

    return images, tf.reshape(label_index_batch, [batch_size])

def image_preprocessing(image, train, resize_size, crop_size, mean=None, thread_id=0):
  """Decode and preprocess one image for evaluation or training.

  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  image = tf.image.resize_images(
    image, [resize_size, resize_size])

  if mean is not None:
    image = tf.subtract(image, mean)

  if train:
    image = distort_image(image, crop_size, crop_size, thread_id)
  else:
    image = eval_image(image, crop_size, crop_size)

  if mean is not None:
    image = tf.multiply(image, 128.)
  else:
    image = tf.image.per_image_standardization(image)
  return image

def distort_image(image, height, width, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.name_scope(values=[image, height, width, bbox], name=scope,
                     default_name='distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # Display the bounding box in the first thread only.

    distorted_image = tf.random_crop(image, [height,width,3])

    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)


    # Randomly distort the colors.
    if FLAGS.distort_color:
      distorted_image = distort_color(distorted_image, thread_id)

    return distorted_image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(values=[image, height, width], name=scope, default_name='eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    return image