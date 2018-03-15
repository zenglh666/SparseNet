import caffe
import numpy as np

caffe.set_mode_gpu()
solver = caffe.SGDSolver('./solver_vgg.prototxt')
solver.net.copy_from('./VGG_jpg.caffemodel')
solver.step(1)
layer_all={
0:'conv1_1',1:'conv1_2',2:'conv2_1',3:'conv2_2',4:'conv3_1',5:'conv3_2',6:'conv3_3',7:'conv4_1',
8:'conv4_2',9:'conv4_3',10:'conv5_1',11:'conv5_2',12:'conv5_3',13:'fc6',14:'fc7',15:'fc8'}
'''
layer_all=(
    'conv1_1','conv2_1','conv3_1','conv3_2',
    'conv4_1','conv4_2','conv5_1','conv5_2',
    'fc6','fc7','fc8')
'''
weights=[]
for i in range(len(layer_all)):
  layer_weights =[]
  if i < 13:
    layer_weights.append(np.transpose(solver.net.params[layer_all[i]][0].data,(3,2,1,0)))
  else:
    layer_weights.append(np.transpose(solver.net.params[layer_all[i]][0].data,(1,0)))
  print(layer_all[i],np.std(layer_weights[0]))
  layer_weights.append(solver.net.params[layer_all[i]][1].data)
  weights.append(layer_weights)
np.savez('vgg16_parameters.npz', parameters = weights)