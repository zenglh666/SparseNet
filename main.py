import os
import math
import logging
from datetime import datetime
import tensorflow as tf
from vgg_sparse import VGGBaseTrain
from vgg_sparse import VGGSparseTrain
from vgg_sparse import VGGSparseFinetune
from utils import *
from variable_list import VariableList


logger = logging.getLogger(__name__)
formatter = logging.Formatter('''%(asctime)s - %(name)s'''
    ''' - %(levelname)s -: %(message)s''')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)
timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('stage','N','''training stage'''
                           '''B:base network training L:layer-wise training '''
                           '''and pruning after layer-wise training '''
                           '''F:fine-tuning''')
tf.app.flags.DEFINE_string('model', None,
                           '''the directory of file saved.''')
tf.app.flags.DEFINE_boolean('use_summary',False,
                            '''If we use tensorflow summary''')
tf.app.flags.DEFINE_boolean('store_middle_ckpt_npy',False,
                            '''If we store ckpt and npy file during training''')
tf.app.flags.DEFINE_integer('total_step',200000,
                            '''the iteration of base training''')
tf.app.flags.DEFINE_integer('save_step',200000,
                            '''the iteration of base training''')
tf.app.flags.DEFINE_integer('test_step',10000,
                            '''the iteration of base training''')


tf.app.flags.DEFINE_string('save_dir', 'E:\\tensorflow',
                           '''the directory of file saved.''')
tf.app.flags.DEFINE_string('restore_check_point_file', '',
                           '''the directory of file saved''')

tf.app.flags.DEFINE_string('log_file', timestr+'.log',
                           '''the directory of file saved''')
tf.app.flags.DEFINE_float('reduce_ratio',0.0,
                          '''the ratio of pruning reduce parameter''')
tf.app.flags.DEFINE_float('sparse_ratio',0.0,
                          '''the ratio of pruning sparse parameter''')
tf.app.flags.DEFINE_float('project_ratio',0.0,
                          '''the ratio of pruning project parameter''')


def evaluation(eval1_op, eval5_op, sess, step):
  num_iter = get_validation_num() // FLAGS.test_batch_size
  eval1_sum = 0.
  eval5_sum = 0.
  for eval_step in range(num_iter):
      eval1, eval5 = sess.run([eval1_op, eval5_op])
      eval1_sum += eval1
      eval5_sum += eval5
  accuracy = eval1_sum / num_iter 
  accuracy5 = eval5_sum / num_iter
  logger.info('step: %d, accuracy_top1 = %.4f, accuracy_top5 = %.4f' % 
      (step, accuracy, accuracy5,))

def print_hyper_param(variable_list):
  logger.info('The shape of whole net:')
  for i in range(variable_list.get_block_count()):
    logger.info('layer name: ' + variable_list.get_name(i) +
                ' shape: ' + str(variable_list.get_shape(i)))

def training(sess, global_step, train_op, loss, eval1_op, eval5_op,
             checkpoint_file, saver, merged=None, train_writer=None):
    
    step = sess.run(global_step)
    while step < (FLAGS.total_step + 1):

      if (merged is not None) and (train_writer is not None) and (step % FLAGS.test_step == 0):
        _, loss_value, summary_str, step = sess.run([train_op, loss, merged, global_step])
        train_writer.add_summary(summary_str, step)
      else:
        _, loss_value , step= sess.run([train_op, loss, global_step])

      if step % 100 == 0:
        logger.info('step %d: loss = %.4f' % (step, loss_value))

      if FLAGS.store_middle_ckpt_npy and (step % FLAGS.save_step) == 0:
        saver.save(sess, checkpoint_file, global_step=step)
        logger.info("Model saved in file: %s" % checkpoint_file)

      if step % FLAGS.test_step == 0:
        evaluation(eval1_op, eval5_op, sess, step)

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img

def summary(loss, global_step):
    tf.summary.scalar(loss.op.name, loss)

    grad_list = tf.get_collection('grads')
    var_list = tf.trainable_variables()
    activation_list = tf.get_collection('activations')
    images = tf.get_collection('images')[0]

    for grad in grad_list:
        if grad is not None:
            tf.summary.histogram(grad.op.name + '/gradients', grad)

    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',
                                 group_batch_images(kernels), max_outputs=1)
    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)
            tf.summary.scalar(activation.op.name + '/sparsity', tf.nn.zero_fraction(activation))
    if images is not None:
        images = tf.multiply(images, 0.5)
        images = tf.subtract(images, -0.5)
        tf.summary.image('/images', images)


def train_base(save_dir):

  if FLAGS.model == 'VGG16':
    model = VGGBaseTrain()
  train_op, loss, global_step = train(model)
  eval1_op, eval5_op = eval(model)

  saver = tf.train.Saver(model.get_conv_variable_chain(), max_to_keep=5)
  checkpoint_file = os.path.join(save_dir, 'base.ckpt')
  
  if FLAGS.use_summary:
    summary(loss, global_step)
    merged = tf.summary.merge_all()
    summary_file = os.path.join(save_dir, 'base_summary')
    train_writer = tf.summary.FileWriter(summary_file)
  else:
    merged = None
    train_writer = None

  config=tf.ConfigProto(
      log_device_placement=False, allow_soft_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

  with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if FLAGS.model == 'VGG16':
      model.read_base_parameters('./vgg_parameters', sess)
      logger.info('successfully load parameter from ./vgg_parameters')

    if FLAGS.restore_check_point_file != '':
      checkpoint_file = os.path.join(save_dir, FLAGS.restore_check_point_file)
      saver.restore(sess, checkpoint_file)
      logger.info('successfully load parameter from %s' % checkpoint_file)

    training(sess, global_step, train_op, loss, eval1_op, eval5_op,
             checkpoint_file, saver, merged, train_writer)

    save_path = saver.save(sess, checkpoint_file)
    logger.info("Model saved in file: %s" % save_path)
    coord.request_stop()
    coord.join(threads)

def train_decompose(save_dir):

  total_epoch = 3
  variable_list = None
  for epoch in range(total_epoch):
    tf.reset_default_graph()
    if FLAGS.model == 'VGG16':
      model = VGGSparseTrain(variable_list)

    train_op, loss, global_step = train(model)
    eval1_op, eval5_op = eval(model, reuse=False)
    saver = tf.train.Saver(model.get_sparse_variable_chain(), max_to_keep=5)
    checkpoint_file = os.path.join(save_dir,'train_%d.ckpt' % epoch)

    if FLAGS.use_summary:
      summary(loss, global_step)
      merged = tf.summary.merge_all()
      summary_file = os.path.join(save_dir, 'train_summary')
      train_writer = tf.summary.FileWriter(summary_file)
    else:
      merged = None
      train_writer = None

    config=tf.ConfigProto(
      log_device_placement=False, allow_soft_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Session(config=config) as sess:
      if epoch == 0:
        logger.info('FLOPS = %d, compression rate = %.4f' % model.calculate_flops())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model.get_conv_variable_chain())
        checkpoint_file = os.path.join(save_dir, '..', 'base.ckpt')
        saver.restore(sess, checkpoint_file)
      else:
        sess.run(tf.global_variables_initializer())
        model.restore_parameters(sess)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      variable_list = model.get_variable_list()
      print_hyper_param(variable_list)
      training(sess, global_step, train_op, loss, eval1_op, eval5_op,
               checkpoint_file, saver, merged, train_writer)
      model.record_parameters(sess)

      if FLAGS.store_middle_ckpt_npy:
        save_path = saver.save(sess, checkpoint_file)

        save_npy_file = os.path.join(save_dir, 'train_%d' % epoch)
        variable_list = model.get_variable_list()
        variable_list.write(save_npy_file)
        logger.info("Model saved in file: %s.npy" % save_npy_file)

      if epoch == 0:
        variable_list = model.pruning_reduce(FLAGS.reduce_ratio)
      elif epoch == 1:
        variable_list = model.pruning_sparse(FLAGS.sparse_ratio)
      else:
        variable_list = model.get_variable_list()

      coord.request_stop()
      coord.join(threads)

  logger.info('FLOPS = %d, compression rate = %.4f' % model.calculate_flops(variable_list))
  save_npy_file = os.path.join(save_dir, 'train')
  variable_list.write(save_npy_file)

def finetune_decompose(save_dir):

  total_epoch = 2

  for epoch in range(total_epoch):

    if epoch == 0:
      save_npy_file = os.path.join(save_dir,'train')
      variable_list = VariableList(file_name = save_npy_file)

    if FLAGS.model == 'VGG16':
      model = VGGSparseFinetune(variable_list)

    if epoch == 0:
      variable_list = model.pruning_project(FLAGS.project_ratio)
    else:
      train_op, loss, global_step = train(model)
      eval1_op, eval5_op = eval(model)
      saver = tf.train.Saver(model.get_sparse_variable_chain(), max_to_keep=5)
      checkpoint_file = os.path.join(save_dir,'cifar_finetune_%d.ckpt' % epoch)

      if FLAGS.use_summary:
        summary(loss, global_step)
        merged = tf.summary.merge_all()
        summary_file = os.path.join(save_dir, 'finetune_summary')
        train_writer = tf.summary.FileWriter(summary_file)
      else:
        merged = None
        train_writer = None

      config=tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True))

      with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        model.restore_parameters(sess)

        variable_list = model.get_variable_list()
        print_hyper_param(variable_list)
        training(sess, global_step, train_op, loss, eval1_op, eval5_op,
                 checkpoint_file, saver, merged, train_writer)
        model.record_parameters(sess)

        if FLAGS.store_middle_ckpt_npy:
          save_path = saver.save(sess, checkpoint_file)

          save_npy_file = os.path.join(save_dir,'cifar_finetune')
          variable_list.write(save_npy_file)
          logger.info("Model saved in file: %s" % save_path)

        variable_list = model.get_variable_list()
        coord.request_stop()
        coord.join(threads)
        
    logger.info('FLOPS = %d, compression rate = %.4f' % model.calculate_flops(variable_list))

  save_npy_file = os.path.join(save_dir,'cifar_finetune')
  variable_list.write(save_npy_file)

def create_dir():
  base_dir = FLAGS.save_dir
  setting_dir = 'rr_%.2f_sr_%.2f_pr_%.2f' % (
    FLAGS.reduce_ratio, FLAGS.sparse_ratio, FLAGS.project_ratio)
  log_dir=os.path.join(base_dir,FLAGS.model,setting_dir)
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  return log_dir

def main(argv=None):  # pylint: disable=unused-argument
    for key, value in FLAGS.__flags.items():
        logger.info('%s: %s' % (key, value))
    
    stage = FLAGS.stage
    save_dir = create_dir()
    if stage=='B':
      train_base(save_dir)
    elif stage=='L':
      train_decompose(save_dir)
    elif stage=='F':
      finetune_decompose(save_dir)
    else:
      logger.info('unkown stage')

if __name__ == "__main__":
  save_dir = create_dir()
  logger.setLevel(logging.DEBUG)
  handler = logging.FileHandler(os.path.join(save_dir,FLAGS.log_file), 'w')
  handler.setLevel(logging.DEBUG)
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  tf.app.run()
  