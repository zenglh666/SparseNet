import tensorflow as tf


class Layers(object):
  def __init__(
      self, weight_init = None, regularizer = None, bias_init = None):
    self.default_regularizer = regularizer
    self.default_weight_init = weight_init
    self.default_bias_init = bias_init
    self.conv_variables = list()
    self.fully_variables = list()
    self.sparse_variables = list()

  def _weight_variable(self, shape, name, trainable = True, init_in = None, regularizer_in = None):
    if init_in == None:
      init = self.default_weight_init
    else:
      init = init_in

    if regularizer_in == None:
      regularizer = self.default_regularizer
    else:
      regularizer = regularizer_in
    var = tf.get_variable(
        shape = shape,
        initializer = init, regularizer = regularizer,
        name = name, trainable = trainable)
    return var

  def _bias_variable(self, shape, name, trainable = True, init_in = None):
    if init_in == None:
      init = self.default_bias_init
    else:
      init = init_in
    var = tf.get_variable(
        shape = shape, initializer = init,
        name = name, trainable = trainable)
    return var

  def conv2d(self, x, variable_scope, shape_list,  trainable = True, reuse = False,
              weights_init = None, bias_init = None, regularizer = None):

    assert type(shape_list) == list
    assert len(shape_list) == 2
    for i in range(2):
      assert type(shape_list[i]) == list
    assert len(shape_list[0]) == 4
    assert len(shape_list[1]) == 1

    with tf.variable_scope(variable_scope, values=[x], reuse=reuse):
      W = self._weight_variable(
        shape_list[0], 'weights', trainable,
        weights_init, regularizer)
      b = self._bias_variable(
        shape_list[1], 'bias', trainable, bias_init)

      if not reuse:
        block_list = list()
        block_list.append(W)
        block_list.append(b)
        self.conv_variables.append(block_list)

      a = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      return tf.nn.bias_add(a, b)

  def conv_sparse(self, x, variable_scope, shape_list, trainable=True, reuse=False,
                   weights_init = None, bias_init = None, regularizer = None):

    assert type(shape_list) == list
    assert len(shape_list) == 6
    for i in range(6):
      assert type(shape_list[i]) == list
    for i in range(5):
      assert len(shape_list[i]) == 4
    assert len(shape_list[5]) == 1

    with tf.variable_scope(variable_scope, values=[x], reuse=reuse):
      W_reduce = self._weight_variable(shape_list[0], 'weights_reduce',
                                       trainable, weights_init, regularizer)
      W_3x1 = self._weight_variable(shape_list[1], 'weights_3x1',
                                    trainable, weights_init, regularizer)
      W_1x3 = self._weight_variable(shape_list[2], 'weights_1x3',
                                    trainable, weights_init, regularizer)
      W_2x2 = self._weight_variable(shape_list[3], 'weights_2x2',
                                    trainable, weights_init, regularizer)
      W_project = self._weight_variable(shape_list[4], 'weights_project',
                                        trainable, weights_init, regularizer)
      b = self._bias_variable(shape_list[5], 'bias', bias_init)

      if not reuse:
        block_list= list()
        block_list.append(W_reduce)
        block_list.append(W_3x1)
        block_list.append(W_1x3)
        block_list.append(W_2x2)
        block_list.append(W_project)
        block_list.append(b)
        self.sparse_variables.append(block_list)

      a_reduce = tf.nn.conv2d(x, W_reduce, strides=[1, 1, 1, 1], padding='SAME')
      a_3x1 = tf.nn.conv2d(a_reduce, W_3x1, strides=[1, 1, 1, 1], padding='SAME')
      a_1x3 = tf.nn.conv2d(a_reduce, W_1x3, strides=[1, 1, 1, 1], padding='SAME')
      a_2x2 = tf.nn.convolution(a_reduce, W_2x2, strides=[1,1], padding='SAME',
                                dilation_rate=[2,2])
      a_concat = tf.concat([a_3x1, a_1x3, a_2x2], 3)
      a_proj = tf.nn.conv2d(a_concat, W_project, strides=[1, 1, 1, 1], padding='SAME')
      return tf.nn.bias_add(a_proj, b)

  def fullyconnect(self, x, variable_scope, shape_list, trainable=True, reuse=False,
                    weights_init = None, bias_init = None, regularizer = None):

    assert type(shape_list) == list
    assert len(shape_list) == 2
    for i in range(2):
      assert type(shape_list[i]) == list
    assert len(shape_list[0]) == 2
    assert len(shape_list[1]) == 1

    with tf.variable_scope(variable_scope, values=[x], reuse=reuse):
      W = self._weight_variable(shape_list[0], 'weights',
                                trainable, weights_init, regularizer)
      b = self._bias_variable(shape_list[1], 'bias', trainable, bias_init)

      if not reuse:
        block_list = list()
        block_list.append(W)
        block_list.append(b)
        self.fully_variables.append(block_list)

      x_flat = tf.reshape(x, [-1, shape_list[0][0]])
      a = tf.matmul(x_flat, W)
      return tf.nn.bias_add(a, b)
    
  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                padding='SAME')