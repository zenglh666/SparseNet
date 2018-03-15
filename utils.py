import tensorflow as tf
import os
import image_reader
from imagenet_data import ImagenetData
from cifar_data import Cifar10Data, Cifar100Data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'None',
                           '''the dataset to be used.''')
tf.app.flags.DEFINE_integer('train_batch_size', 8,
                            """eval examples num.""")
tf.app.flags.DEFINE_integer('test_batch_size', 8,
                            """eval examples num.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('grad_clip_norm', 1e2,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_string('optimizer', 'SGD',
                            """optimizer.""")
tf.app.flags.DEFINE_integer('num_gpu', 1,
                            """number of gpu.""")
tf.app.flags.DEFINE_integer('decay_iter', 50000,
                            """Iterations after which learning rate decays.""")
tf.app.flags.DEFINE_boolean('debug',False,
                            '''If debug''')


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad = tf.clip_by_norm(grad, FLAGS.grad_clip_norm, name=v.op.name+'_grad')

      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

def get_validation_num():
  if FLAGS.dataset == 'imagenet':
    return ImagenetData('imagenet', subset='validation').num_examples_per_epoch()
  elif FLAGS.dataset == 'cifar10':
    return Cifar10Data('cifar10', subset='validation').num_examples_per_epoch()
  elif FLAGS.dataset == 'cifar100':
    return Cifar100Data('cifar100', subset='validation').num_examples_per_epoch()


def inputs(subset):
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if FLAGS.dataset is None:
      raise ValueError('Please supply a dataset')

    if FLAGS.dataset == 'imagenet':
      dataset = ImagenetData('imagenet', subset=subset)
    elif FLAGS.dataset == 'cifar10':
      dataset = Cifar10Data('cifar10', subset=subset)
    elif FLAGS.dataset == 'cifar100':
      dataset = Cifar100Data('cifar100', subset=subset)

    if subset == 'train':
      images, labels = image_reader.create_data_batch(dataset, FLAGS.train_batch_size)
    elif subset == 'validation':
      images, labels = image_reader.create_data_batch(dataset, FLAGS.test_batch_size)

    tf.add_to_collection('images', images)
    return images, labels

def train(model):
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        FLAGS.initial_learning_rate, global_step, FLAGS.decay_iter,
        FLAGS.learning_rate_decay_factor, staircase=True)

    if FLAGS.optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(lr, 0.9)
    else:
        raise ValueError('optimizer unsupported')

    assert FLAGS.train_batch_size % FLAGS.num_gpu == 0, (
        'Batch size must be divisible by number of GPUs')

    images, labels = inputs('train')
    images_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=images)
    labels_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=labels)
      
    tower_grads = []
    reuse = False
    for i in range(FLAGS.num_gpu):
      if FLAGS.debug:
        devicestr = '/gpu:0'
      else:
        devicestr = '/gpu:'+str(i)
      with tf.device(devicestr):
        with tf.name_scope('%s_%d' % ('train', i)) as scope:
          loss = model.loss(images_splits[i], labels_splits[i], reuse=reuse)
          tf.add_to_collection('losses', loss)
          reuse = True
          trainable_variables = tf.trainable_variables()
          grads = tf.gradients(loss, trainable_variables)
          grads_vars = zip(grads, trainable_variables)
          tower_grads.append(grads_vars)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for grad, var in grads:
      tf.add_to_collection('grads', grad)

    loss = tf.reduce_sum(tf.get_collection('losses'), name='loss')

    ema = tf.train.ExponentialMovingAverage(
        0.997, global_step, name='average')
    ema_op = ema.apply([loss])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.group(*updates_collection)

    return train_op, loss, global_step

def eval(model, reuse = True):
    assert FLAGS.test_batch_size % FLAGS.num_gpu == 0, (
        'Batch size must be divisible by number of GPUs')
    images, labels = inputs('validation')
    images_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=images)
    labels_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=labels)
    for i in range(FLAGS.num_gpu):
      with tf.name_scope('%s_%d' % ('test', i)) as scope:
        if FLAGS.debug:
          devicestr = '/gpu:0'
        else:
          devicestr = '/gpu:'+str(i)
        with tf.device(devicestr):
          logits = model.inference(images_splits[i], 'validation', reuse=reuse)
          reuse = True
        acc1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_splits[i], 1), tf.float32))
        acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_splits[i], 5), tf.float32))
        tf.add_to_collection('acc1', acc1)
        tf.add_to_collection('acc5', acc5)
    total_acc1 = tf.reduce_mean(tf.get_collection('acc1'))
    total_acc5 = tf.reduce_mean(tf.get_collection('acc5'))
    return total_acc1, total_acc5