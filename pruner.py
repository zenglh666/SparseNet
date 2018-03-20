import numpy as np
from sklearn.decomposition import PCA

class PCAL2Pruner(object):
  def _rank_selection(self, block_param_front, ratio):
    block_count = len(block_param_front)

    block_pruning_count = list()
    for i in range(block_count):
      cell_param_front = block_param_front[i]
      cell_count = len(cell_param_front)
      cell_pruning_count = list()

      for j in range(cell_count):
        if cell_param_front[j].shape[3] > 1:
          kernel_count = cell_param_front[j].shape[3]
          param_reshape=np.reshape(cell_param_front[j], (-1, kernel_count))
          pca = PCA(n_components = kernel_count)
          pca.fit(param_reshape)
          keep = (pca.explained_variance_ratio_ /
              np.average(pca.explained_variance_ratio_))

          pruning_count = np.sum(keep < ratio)

          if pruning_count < kernel_count:
            cell_pruning_count.append(pruning_count)
          else:
            cell_pruning_count.append(0)
        else:
          cell_pruning_count.append(0)
      block_pruning_count.append(cell_pruning_count)
    return block_pruning_count

  def check_legal(self, block_param_front, 
          block_param_after):
    assert type(block_param_front) == list
    assert type(block_param_after) == list
    assert (len(block_param_front) == len(block_param_after))

    block_count = len(block_param_front)
    for i in range(block_count):
      cell_param_front = block_param_front[i]
      cell_param_after = block_param_after[i]

      assert type(cell_param_front) == list
      assert type(cell_param_after) == list

      output_channel = 0
      for j in range(len(cell_param_front)):
        assert type(cell_param_front[j]) == np.ndarray

        if j == 0:
          input_channel = cell_param_front[j].shape[2]
        else:
          assert (cell_param_front[j].shape[2] == input_channel)

        output_channel += cell_param_front[j].shape[3]
      for j in range(len(cell_param_after)):
        assert type(cell_param_after[j]) == np.ndarray
        assert (cell_param_after[j].shape[2] == output_channel)

  def pruning(self, block_param_front, block_param_after, ratio,
              block_param_front_b = None):
    self.check_legal(block_param_front, block_param_after)

    pruning_count = self._rank_selection(block_param_front, ratio)

    block_count = len(block_param_front)
    pruned_block_param_front = list()
    pruned_block_param_after = list()
    if block_param_front_b != None:
      pruned_block_param_front_b = list()

    for i in range(block_count):
      retain_front = np.empty(0, dtype = np.int)
      retain_after = np.empty(0, dtype = np.int)
      retain_num = 0

      cell_param_front = block_param_front[i]
      cell_param_after = block_param_after[i]
      pruned_cell_param_front = list()
      pruned_cell_param_after = list()

      if block_param_front_b != None:
        cell_param_front_b = block_param_front_b[i]
        pruned_cell_param_front_b = list()

      for j in range(len(cell_param_front)):
        shape = list(cell_param_front[j].shape)
        param_front_reshape = np.reshape(cell_param_front[j], (-1, shape[3]))

        layer_norm = np.linalg.norm(param_front_reshape, axis = 0)
        norm_sort_inc=np.argsort(layer_norm)
        retain = norm_sort_inc[pruning_count[i][j]:]
        retain = np.sort(retain)

        shape[3] -= pruning_count[i][j]
        pruned_cell_param_front.append(
          np.zeros(shape, cell_param_front[j].dtype))

        for k in range(shape[3]):
          pruned_cell_param_front[j][:,:,:,k] = cell_param_front[j][:,:,:,retain[k]]

        if block_param_front_b != None:
          pruned_cell_param_front_b.append(np.zeros([shape[3]], cell_param_front[j].dtype))
          pruned_cell_param_front_b[j][:] = cell_param_front_b[j][retain]

        retain += np.int(retain_num)
        retain_num += shape[3]
        retain_after = np.concatenate((retain_after,retain))

      for j in range(len(cell_param_after)):
        shape = list(cell_param_after[j].shape)
        shape[2] = retain_num

        pruned_cell_param_after.append(np.zeros(shape, cell_param_after[j].dtype))
        for k in range(retain_num):
          pruned_cell_param_after[j][:,:,k,:] = cell_param_after[j][:,:,retain_after[k],:]

      pruned_block_param_front.append(pruned_cell_param_front)
      pruned_block_param_after.append(pruned_cell_param_after)

      if block_param_front_b != None:
        pruned_block_param_front_b.append(pruned_cell_param_front_b)

    if block_param_front_b != None:
      return (pruned_block_param_front, pruned_block_param_after,
              pruned_block_param_front_b)
    return (pruned_block_param_front, pruned_block_param_after)
