import tensorflow as tf
import numpy as np
import copy
from variable_list import VariableList
from layers import Layers
from pruner import PCAL2Pruner

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('regularizer_factor',0.0005,
                          '''the ratio of pruning project parameter''')

class Model():
  def __init__(self, variable_list, fully_index, sparse_index, sparse_num, input_shape):

    self._variable_list = copy.deepcopy(variable_list)
    self.layers = Layers(
      weight_init=tf.contrib.layers.xavier_initializer(),
      regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularizer_factor),
      bias_init=tf.constant_initializer(0.0))
    self.fully_index = fully_index
    self.sparse_index = sparse_index
    self.sparse_num =  sparse_num
    self.index_fix = [0, self.fully_index, self.sparse_index]
    self._input_shape = input_shape

  def read_base_parameters(self, filename, sess):
    self._variable_list.read_parameters(filename)
    block_list = (
      self.layers.conv_variables + self.layers.fully_variables)
    assing_list = []
    for i in range(len(block_list)):
      for j in range(len(block_list[i])):
        parameter = self._variable_list.get_parameter(i, j)
        variable = block_list[i][j]
        assing_list.append(variable.assign_add(parameter))
    sess.run(assing_list)

  def record_parameters(self, sess):
    block_list = (
      self.layers.conv_variables, self.layers.fully_variables, self.layers.sparse_variables)

    for i in range(3):
      for j in range(len(block_list[i])):
        for k in range(len(block_list[i][j])):
          parameter = sess.run(block_list[i][j][k])
          self._variable_list.set_parameter(self.index_fix[i] + j, k, parameter)

  def restore_parameters(self, sess):
    block_list = (
      self.layers.conv_variables, self.layers.fully_variables, self.layers.sparse_variables)

    assing_list = []
    for i in range(3):
      for j in range(len(block_list[i])):
        for k in range(len(block_list[i][j])):
          parameter = self._variable_list.get_parameter(self.index_fix[i] + j, k)
          variable = block_list[i][j][k]
          assing_list.append(variable.assign(parameter))
    sess.run(assing_list)

  def get_variable_list(self):
    return copy.deepcopy(self._variable_list)

  def get_conv_variable_chain(self):
    convolution_flattened = list(variable 
      for sublist in self.layers.conv_variables
      for variable in sublist)
    fullyconnect_flattened = list(variable 
      for sublist in self.layers.fully_variables
      for variable in sublist)
    return convolution_flattened + fullyconnect_flattened

  def get_sparse_variable_chain(self):
    convolution_flattened = list(variable 
      for sublist in self.layers.conv_variables
      for variable in sublist)
    fullyconnect_flattened = list(variable 
      for sublist in self.layers.fully_variables
      for variable in sublist)
    sparse_flattend = list(variable 
      for sublist in self.layers.sparse_variables
      for variable in sublist)
    return (convolution_flattened + fullyconnect_flattened +
        sparse_flattend)

  def pruning_reduce(self, ratio):
    variable_list = copy.deepcopy(self._variable_list)

    block_param_front = list()
    block_param_after = list()

    for i in range(self.sparse_num):

      cell_param_front = list()
      cell_param_front.append(variable_list.get_parameter(i + self.sparse_index, 0))
      block_param_front.append(cell_param_front)

      cell_param_after = list()
      for j in range(3):
        cell_param_after.append(variable_list.get_parameter(i + self.sparse_index, j + 1))
      block_param_after.append(cell_param_after)

    pruner = PCAL2Pruner()
    pruned_block_param_front, pruned_block_param_after = pruner.pruning(
      block_param_front, block_param_after, ratio)

    for i in range(self.sparse_num):
      variable_list.set_shape(i + self.sparse_index, 0, list(pruned_block_param_front[i][0].shape))
      variable_list.set_parameter(i + self.sparse_index, 0, pruned_block_param_front[i][0])

      for j in range(3):
        variable_list.set_shape(
          i + self.sparse_index, j + 1, list(pruned_block_param_after[i][j].shape))
        variable_list.set_parameter(i + self.sparse_index, j + 1, pruned_block_param_after[i][j])

    return variable_list

  def pruning_sparse(self, ratio):
    variable_list = copy.deepcopy(self._variable_list)
    block_param_front = list()
    block_param_after = list()
    
    for i in range(self.sparse_num):

      cell_param_after = list()
      cell_param_after.append(variable_list.get_parameter(i + self.sparse_index, 4))
      block_param_after.append(cell_param_after)

      cell_param_front = list()
      for j in range(3):
        cell_param_front.append(
          variable_list.get_parameter(i + self.sparse_index, j + 1))
      block_param_front.append(cell_param_front)

    pruner = PCAL2Pruner()
    pruned_block_param_front, pruned_block_param_after = pruner.pruning(
      block_param_front, block_param_after, ratio)

    for i in range(self.sparse_num):
      variable_list.set_shape(i + self.sparse_index, 4, list(pruned_block_param_after[i][0].shape))
      variable_list.set_parameter(i + self.sparse_index, 4, pruned_block_param_after[i][0])
      for j in range(3):
        variable_list.set_shape(
          i + self.sparse_index, j + 1, list(pruned_block_param_front[i][j].shape))
        variable_list.set_parameter(i + self.sparse_index, j + 1, pruned_block_param_front[i][j])

    return variable_list

  def pruning_project(self, ratio):
    variable_list = copy.deepcopy(self._variable_list)
    block_param_front = list()
    block_param_b_front = list()
    block_param_after = list()

    for i in range(self.sparse_num):
      cell_param_front = list()
      cell_param_front.append(variable_list.get_parameter(i + self.sparse_index, 4))
      block_param_front.append(cell_param_front)
      cell_param_after = list()

      if i < (self.sparse_num - 1):
        cell_param_after.append(variable_list.get_parameter(i + self.sparse_index + 1, 0))
      else:
        parameter = variable_list.get_parameter(self.fully_index, 0)
        parameter_reshspe = np.reshape(parameter, 
          (self._input_shape[-1]//2, self._input_shape[-1]//2, -1, parameter.shape[1]))
        cell_param_after.append(parameter_reshspe)    
      block_param_after.append(cell_param_after)

      cell_param_b_front = list()
      cell_param_b_front.append(variable_list.get_parameter(i + self.sparse_index, 5))
      block_param_b_front.append(cell_param_b_front)

    pruner = PCAL2Pruner()
    pruned_block_param_front, pruned_block_param_after, pruned_block_param_b_front = pruner.pruning(
      block_param_front, block_param_after, ratio, block_param_b_front)
    for i in range(self.sparse_num):
      variable_list.set_shape(i + self.sparse_index, 4, list(pruned_block_param_front[i][0].shape))
      variable_list.set_parameter(i + self.sparse_index, 4, pruned_block_param_front[i][0])

      if i < (self.sparse_num - 1):
        variable_list.set_shape(
          i + self.sparse_index + 1, 0, list(pruned_block_param_after[i][0].shape))
        variable_list.set_parameter(i + self.sparse_index + 1, 0, pruned_block_param_after[i][0])
      else:
        shape = pruned_block_param_after[i][0].shape
        shape_reshape = [shape[0] * shape[1] * shape[2], shape[3]]

        variable_list.set_shape(self.fully_index, 0, shape_reshape)
        parameter = pruned_block_param_after[i][0]
        parameter_reshape = np.reshape(parameter, shape_reshape)
        variable_list.set_parameter(self.fully_index, 0, parameter_reshape)

      variable_list.set_shape(
        i + self.sparse_index, 5, list(pruned_block_param_b_front[i][0].shape))
      variable_list.set_parameter(i + self.sparse_index, 5, pruned_block_param_b_front[i][0])
    return variable_list