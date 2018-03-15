from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from variable_list import VariableList
from model import Model

FLAGS = tf.app.flags.FLAGS

class VGGModel(Model):
  _CONV_NAMES = (
    'conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',
    'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3')
  _SPARSE_NAMES = (
    'sparse1_2','sparse2_1','sparse2_2','sparse3_1','sparse3_2',
    'sparse3_3','sparse4_1','sparse4_2','sparse4_3','sparse5_1','sparse5_2',
    'sparse5_3')

  _FULLY_NAMES = ('fc1','fc2','fc3')

  _CONV_SHAPES_INIT = [
    [[3,3,3,64],[64]],
    [[3,3,64,64],[64]],
    [[3,3,64,128],[128]],
    [[3,3,128,128],[128]],
    [[3,3,128,256],[256]],
    [[3,3,256,256],[256]],
    [[3,3,256,256],[256]],
    [[3,3,256,512],[512]],
    [[3,3,512,512],[512]],
    [[3,3,512,512],[512]],
    [[3,3,512,512],[512]],
    [[3,3,512,512],[512]],
    [[3,3,512,512],[512]]]

  _SPARSE_SHAPES_INIT = [
    [[1,1,64,64],[3,1,64,24],[1,3,64,24],[2,2,64,24],[1,1,72,64],[64]],
    [[1,1,64,64],[3,1,64,48],[1,3,64,48],[2,2,64,48],[1,1,144,128],[128]],
    [[1,1,128,144],[3,1,144,48],[1,3,144,48],[2,2,144,48],[1,1,144,128],[128]],
    [[1,1,128,144],[3,1,144,96],[1,3,144,96],[2,2,144,96],[1,1,288,256],[256]],
    [[1,1,256,288],[3,1,288,96],[1,3,288,96],[2,2,288,96],[1,1,288,256],[256]],
    [[1,1,256,288],[3,1,288,96],[1,3,288,96],[2,2,288,96],[1,1,288,256],[256]],
    [[1,1,256,288],[3,1,288,192],[1,3,288,192],[2,2,288,192],[1,1,576,512],[512]],
    [[1,1,512,576],[3,1,576,192],[1,3,576,192],[2,2,576,192],[1,1,576,512],[512]],
    [[1,1,512,576],[3,1,576,192],[1,3,576,192],[2,2,576,192],[1,1,576,512],[512]],
    [[1,1,512,576],[3,1,576,192],[1,3,576,192],[2,2,576,192],[1,1,576,512],[512]],
    [[1,1,512,576],[3,1,576,192],[1,3,576,192],[2,2,576,192],[1,1,576,512],[512]],
    [[1,1,512,576],[3,1,576,192],[1,3,576,192],[2,2,576,192],[1,1,576,512],[512]]]

  _FULLY_SHAPES_INIT = [
    [[512*7*7,4096],[4096]],
    [[4096,4096],[4096]],
    [[4096,1000],[1000]]]

  _repeat_num = [2,2,3,3,3]
  _loss_weights = [1,1,1,1,1,1,1,1,1,1,10,10]
  _input_shape = [224, 224, 112, 112, 56, 56, 56, 28, 28, 28 ,14, 14, 14]

  def __init__(self, variable_list=None, data_type='float32'):
    self._data_type = data_type
    if variable_list != None:
      variable_list_in = variable_list
    else:
      shape_list = list(self._CONV_SHAPES_INIT + self._FULLY_SHAPES_INIT +
                        self._SPARSE_SHAPES_INIT)
      name_list = list(self._CONV_NAMES + self._FULLY_NAMES + self._SPARSE_NAMES)
      variable_list_in = VariableList(shape_list, name_list, data_type)

    fully_index = len(self._CONV_NAMES)
    sparse_index = len(self._CONV_NAMES) + len(self._FULLY_NAMES)
    sparse_num =  len(self._SPARSE_NAMES)
    super().__init__(variable_list_in,
                     fully_index, sparse_index, sparse_num, self._input_shape)

  def calculate_flops(self, variable_list = None):
    if variable_list == None:
      variable_list = self._variable_list
    input_shape = self._input_shape

    flops_sum = 0
    for i in range(self.sparse_num):
      if i==0:
        shape = self._CONV_SHAPES_INIT[i]
      else:
        shape = variable_list.get_shape(self._SPARSE_NAMES[i - 1])

      for j in range(len(shape) - 1):
        flops = input_shape[i] * input_shape[i]
        for k in shape[j]:
          flops *= k
        flops_sum += flops

    origin_flops_sum = 0
    for i in range(len(self._CONV_NAMES)):
      shape = self._CONV_SHAPES_INIT[i]
      for j in range(len(shape) - 1):
        flops = input_shape[i] * input_shape[i]
        for k in shape[j]:
          flops *= k
        origin_flops_sum += flops 
    return flops_sum, float(origin_flops_sum) / float (flops_sum)






class VGGBaseTrain(VGGModel):
  def __init__(self, variable_list=None, data_type='float32'):
    super().__init__(variable_list, data_type)

  def inference(self, x, stage, reuse = False):
      conv_layer_count = 0
      layer_input = x
      layers = self.layers
      for i in range(len(self._repeat_num)):

        for j in range(self._repeat_num[i]):
          a_conv = layers.conv2d(layer_input,
            self._variable_list.get_name(conv_layer_count),
            self._variable_list.get_shape(conv_layer_count),
            reuse = reuse)
          tf.add_to_collection('activations', a_conv)
          conv_layer_count += 1
          h_conv = tf.nn.relu(a_conv)
          layer_input = h_conv

        h_pool = tf.nn.max_pool(
          layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer_input = h_pool

      h_pool_flat = tf.reshape(
        h_pool, [-1, self._variable_list.get_shape(conv_layer_count)[0][0]])

      a_fc1 = layers.fullyconnect(h_pool_flat,
        self._variable_list.get_name(conv_layer_count),
        self._variable_list.get_shape(conv_layer_count),
        reuse = reuse)
      tf.add_to_collection('activations', a_fc1)
      h_fc1 = tf.nn.relu(a_fc1)

      if stage == 'train':
        h_fc1 = tf.nn.dropout(h_fc1, 0.5)

      a_fc2 = layers.fullyconnect(h_fc1,
        self._variable_list.get_name(conv_layer_count + 1),
        self._variable_list.get_shape(conv_layer_count + 1),
        reuse = reuse)
      tf.add_to_collection('activations', a_fc2)
      h_fc2 = tf.nn.relu(a_fc2)

      if stage == 'train':
        h_fc2 = tf.nn.dropout(h_fc2, 0.5)

      a_fc3 = layers.fullyconnect(h_fc2,
        self._variable_list.get_name(conv_layer_count + 2),
        self._variable_list.get_shape(conv_layer_count + 2),
        reuse = reuse)
      tf.add_to_collection('activations', a_fc3)
      return a_fc3

  def loss(self, images, labels, stage='train', reuse=False):
      logits = self.inference(images, stage, reuse)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
      tf.add_to_collection('cross_entropy_losses', cross_entropy)
      regularizer_loss = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
        name='regularize_loss')
      total_loss = tf.add(cross_entropy, regularizer_loss,
        name='total_loss')
      return total_loss







class VGGSparseTrain(VGGModel):

  def __init__(self, variable_list = None, data_type = 'float32'):
    super().__init__(variable_list, data_type)

  def loss(self, x, stage='train', reuse=False):
      conv_layer_count = 0
      conv_layer_list = list()
      conv_layer_input_list = list()
      layer_input = x

      for i in range(len(self._repeat_num)):
        for j in range(self._repeat_num[i]):
          conv_layer_input_list.append(layer_input)
          a_conv = self.layers.conv2d(layer_input,
            self._variable_list.get_name(conv_layer_count),
            self._variable_list.get_shape(conv_layer_count),
            reuse = reuse,trainable = False)
          tf.add_to_collection('activations', a_conv)

          conv_layer_count += 1
          conv_layer_list.append(a_conv)

          if (i != (len(self._repeat_num)-1)) or (j != (self._repeat_num[i]-1)):
            h_conv = h_conv1=tf.nn.relu(a_conv)
            layer_input = h_conv

        if i != (len(self._repeat_num)-1):
          h_pool = tf.nn.max_pool(
            layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
          layer_input = h_pool

      sparse_layer_count = 0
      for i in range(self.sparse_num):
        a_sparse = self.layers.conv_sparse(conv_layer_input_list[i + 1],
          self._variable_list.get_name(self.sparse_index + i),
          self._variable_list.get_shape(self.sparse_index + i),
          reuse = reuse)
        tf.add_to_collection('activations', a_sparse)
        euc_loss = self._loss_weights[i]*tf.reduce_mean(
          tf.square(tf.subtract(conv_layer_list[i + 1], a_sparse)))
        tf.add_to_collection('euclidean_losses_per_layer', euc_loss)
        sparse_layer_count+=1

      euclidean_loss = tf.reduce_mean(
        tf.get_collection('euclidean_losses_per_layer'), name='euclidean')
      tf.add_to_collection('euclidean_losses', euclidean_loss)
      regularizer_loss = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
        name='regularize_losses')
      total_loss = tf.add(euclidean_loss, regularizer_loss, name='total_loss')
      return total_loss

  def inference(self, x, stage, reuse = False):
    sparse_layer_count = 0
    layer_input = x

    for i in range(len(self._repeat_num)):
      for j in range(self._repeat_num[i]):

        if i==0 and j==0:
          a_conv = self.layers.conv2d(layer_input, self._variable_list.get_name(0),
            self._variable_list.get_shape(0), reuse = True)
        else:
          a_conv = self.layers.conv_sparse(layer_input,
            self._variable_list.get_name(self.sparse_index + sparse_layer_count),
            self._variable_list.get_shape(self.sparse_index + sparse_layer_count),
            reuse = True)
          tf.add_to_collection('activations', a_conv)
          sparse_layer_count += 1

        h_conv = tf.nn.relu(a_conv)
        layer_input = h_conv

      h_pool = tf.nn.max_pool(
        layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      layer_input = h_pool

    h_pool_flat = tf.reshape(
      h_pool, [-1, self._variable_list.get_shape(self.fully_index)[0][0]])

    a_fc1 = self.layers.fullyconnect(h_pool_flat, self._variable_list.get_name(self.fully_index),
      self._variable_list.get_shape(self.fully_index), reuse = reuse, trainable = False)
    tf.add_to_collection('activations', a_fc1)

    h_fc1 = tf.nn.relu(a_fc1)
    if stage =='train':
      h_fc1 = tf.nn.dropout(h_fc1, 0.5)

    a_fc2 = self.layers.fullyconnect(h_fc1, self._variable_list.get_name(self.fully_index + 1),
      self._variable_list.get_shape(self.fully_index + 1), reuse=reuse, trainable = False)
    tf.add_to_collection('activations', a_fc2)

    h_fc2 = tf.nn.relu(a_fc2)
    if  stage =='train':
      h_fc2 = tf.nn.dropout(h_fc2, 0.5)

    a_fc3 = self.layers.fullyconnect(h_fc2, self._variable_list.get_name(self.fully_index + 2),
      self._variable_list.get_shape(self.fully_index + 2), reuse=reuse, trainable = False)
    tf.add_to_collection('activations', a_fc3)
    return a_fc3






class VGGSparseFinetune(VGGModel):

  def __init__(self, variable_list, data_type = 'float32'):
    super().__init__(variable_list, data_type)

  def inference(self, x, stage, reuse=False):
    sparse_layer_count = 0
    layer_input = x

    for i in range(len(self._repeat_num)):
      for j in range(self._repeat_num[i]):
        if i==0 and j==0:
          a_conv = self.layers.conv2d(layer_input, self._variable_list.get_name(0),
            self._variable_list.get_shape(0), reuse = reuse)
        else:
          a_conv = self.layers.conv_sparse(layer_input, 
            self._variable_list.get_name(self.sparse_index + sparse_layer_count),
            self._variable_list.get_shape(self.sparse_index + sparse_layer_count),
            reuse = reuse)
          tf.add_to_collection('activations', a_conv)

          sparse_layer_count += 1

        h_conv = tf.nn.relu(a_conv)
        layer_input = h_conv

      h_pool = tf.nn.max_pool(
        layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      layer_input = h_pool

    h_pool_flat = tf.reshape(
      h_pool, [-1, self._variable_list.get_shape(self.fully_index)[0][0]])

    a_fc1 = self.layers.fullyconnect(h_pool_flat, self._variable_list.get_name(self.fully_index),
      self._variable_list.get_shape(self.fully_index), reuse = reuse)
    tf.add_to_collection('activations', a_fc1)
    h_fc1 = tf.nn.relu(a_fc1)

    if stage == 'train':
      h_fc1 = tf.nn.dropout(h_fc1, 0.5)

    a_fc2 = self.layers.fullyconnect(h_fc1, self._variable_list.get_name(self.fully_index + 1),
      self._variable_list.get_shape(self.fully_index + 1), reuse=reuse)
    tf.add_to_collection('activations', a_fc2)
    h_fc2 = tf.nn.relu(a_fc2)

    if stage == 'train':
      h_fc2 = tf.nn.dropout(h_fc2, 0.5)

    a_fc3 = self.layers.fullyconnect(h_fc2, self._variable_list.get_name(self.fully_index + 2),
      self._variable_list.get_shape(self.fully_index + 2), reuse=reuse)
    tf.add_to_collection('activations', a_fc3)

    return a_fc3

  def loss(self, images, labels, stage='train', reuse=False):
      logits = self.inference(images, stage, reuse)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
      tf.add_to_collection('cross_entropy_losses', cross_entropy)
      regularizer_loss = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
        name='regularize_loss')
      total_loss = tf.add(cross_entropy, regularizer_loss,
        name='total_loss')
      return total_loss