import numpy as np

class BlockVariable(object):

  def __init__(self, shape_list, data_type, name):
    assert type(name) == str
    self._name = name
    assert type(shape_list) == list
    self._cell_count = len(shape_list)
    assert type(data_type) == str or type(data_type) == np.dtype
    self._dtype = np.dtype(data_type)

    self._cell_shape_list = list()
    self._cell_parameter_list = list()
    for i in range(self._cell_count):
      assert type(shape_list[i]) == list
      for j in range(len(shape_list[i])):
        assert type(shape_list[i][j]) == int
      self._cell_shape_list.append(shape_list[i])
      self._cell_parameter_list.append(
        np.zeros(shape_list[i], self._dtype))

  def __del__(self):
    self._cell_shape_list.clear()
    self._cell_parameter_list.clear()

  def check_legal(self):
    assert len(self._cell_shape_list) == self._cell_count
    assert len(self._cell_parameter_list) == self._cell_count
    for i in range(self._cell_count):
      assert type(self._cell_shape_list[i]) == list
      assert type(self._cell_parameter_list[i]) == np.ndarray
      assert (self._cell_shape_list[i] == list(self._cell_parameter_list[i].shape))

  def set_shape(self, index, shape):
    assert type(shape) == list
    self._cell_shape_list[index] = shape
    self._cell_parameter_list[index] = np.zeros(shape, dtype=self._dtype)

  def get_shape(self, index):
    return list(self._cell_shape_list[index])

  def set_parameter(self, index, parameter):
    assert type(parameter) == np.ndarray
    assert parameter.shape == self._cell_parameter_list[index].shape
    np.copyto(self._cell_parameter_list[index], parameter)

  def get_parameter(self, index):
    return np.copy(self._cell_parameter_list[index])

  def get_cell_count(self):
    return self._cell_count


class VariableList(object):
  def __init__(self, block_shape_list = None,  block_name_list = None,
               data_type = None,  file_name = None):
    if file_name == None:
      assert type(data_type) == str or type(data_type) == np.dtype
      self._dtype = np.dtype(data_type)
      assert type(block_shape_list) == list
      assert type(block_name_list) == list
      assert len(block_shape_list) == len(block_name_list)
      self._block_count = len(block_shape_list)

      for i in range(self._block_count):
        assert type(block_name_list[i]) == str
        assert type(block_shape_list[i]) == list
      self._block_name_list = list()
      self._block_name_index = dict()
      self._block_list = list()

      for i in range(self._block_count):
        self._block_name_list.append(block_name_list[i])
        self._block_name_index[self._block_name_list[i]] = i
        self._block_list.append(BlockVariable(block_shape_list[i],
                                self._dtype, self._block_name_list[i]))
    else:
      self.read(file_name)

  def check_legal(self):
    for i in range(self._block_count):
      self._block_list[i].check_legal()

  def get_name(self, index):
    return self._block_name_list[index]

  def get_parameter(self, index_block, index_cell = None):
    if type(index_block) == str:
      index_block = self._block_name_index[index_block]

    if index_cell == None:
      parameter_list = list()
      block = self._block_list[index_block]

      for i in range(block.get_cell_count()):
        parameter_list.append(block.get_parameter(i))

      return parameter_list
    else:
      return self._block_list[index_block].get_parameter(index_cell)

  def set_parameter(self, index_block, index_cell, parameter):
    if type(index_block) == str:
      index_block = self._block_name_index[index_block]

    self._block_list[index_block].set_parameter(index_cell,parameter)

  def get_shape(self, index_block, index_cell = None):
    if type(index_block) == str:
      index_block = self._block_name_index[index_block]
    if index_cell == None:
      shape_list = list()
      block = self._block_list[index_block]
      for i in range(block.get_cell_count()):
        shape_list.append(block.get_shape(i))
      return shape_list
    else:
      return self._block_list[index_block].get_shape(index_cell)

  def set_shape(self, index_block, index_cell, shape):
    if type(index_block) == str:
      index_block = self._block_name_index[index_block]
    self._block_list[index_block].set_shape(index_cell, shape)

  def get_block_count(self):
    return self._block_count

  def write(self, file_name):
    block_list_save = np.array(self._block_list, dtype = object)
    block_name_list_save = np.array(self._block_name_list, dtype = str)
    block_count_save = np.array([self._block_count], dtype = np.int)
    np.savez(file_name + '.npz', 
         block_list = block_list_save,
         block_name_list = block_name_list_save,
         block_count = block_count_save)

  def read_parameters(self, file_name):
    parameters_list_read = np.load(file_name + '.npz')
    parameters_list = parameters_list_read['parameters']

    for i in range(len(parameters_list)):
      block = self._block_list[i]
      for j in range(block.get_cell_count()):
        block.set_parameter(j, parameters_list[i][j])

  def read(self, file_name):
    variable_list_read = np.load(file_name + '.npz')
    self._block_list = list(variable_list_read['block_list'])
    self._block_name_list = list(variable_list_read['block_name_list'])
    self._block_count = variable_list_read['block_count'][0]
    self._block_name_index = dict()

    for i in range(self._block_count):
      self._block_name_index[self._block_name_list[i]] = i

  def erase(self, i, j):
    del self._block_name_list[i:j]
    del self._block_list[i:j]
    self._block_count -= abs(i - j)
    self._block_name_index = dict()
    
    for i in range(self._block_count):
      self._block_name_index[self._block_name_list[i]] = i
