import tensorflow as tf
import numpy
from gensim.models import Word2Vec
VEC_SIZE = 60
KERNEL_WIDTH = 3
PROPORTION = 0.5
LOCAL_3 = 400
LOCAL_4 = 200
CLASS_NUM = 5
LEARNING_RATE = 0.1
DECAY_STEP = 1000
DECAY_RATE = 0.96
MOVING_AVERAGE_DECAY= 0.1
BASE_DATA_PATH = 'd:/python/op/data'
def generate_static_vector():
    """
    准备好需要的词向量字典，为之后的生成词向量做准备

    由于需要分成两个步骤，静态词向量和可以训练的词向量，所以分成两步
    """

    pass
def generate_variable_vector():
    """
    本部分进行的是可训练的词向量
    :return:
    """
    return
def _activation_summary(x):
    """
    为定义的操作添加总结，加入方便生成模型和流程图
    :param x:
    :return:
    """
    tensorname = x.op.name
    tf.summary.histogram(tensorname+"/activation",x)
    tf.summary.scalar(tensorname+'/sparsity',tf.nn.zero_fraction(x))
def _variable_on_cpu(name,shape,initializer):
    """
    从示例代码里看到的函数方法，目的大概是创建缓存在cpu中的变量
    :param name: 变量名
    :param shape: 变量形状，大小维度
    :param initializer: 变量初始化工具，如随机初始化的truncate_normal
    :return: var 生成变量

    """
    with tf.device('/cpu:0'):
        var = tf.Variable(name=name,shape=shape,dtype=tf.float32,initializer = initializer)
    return var
def _variable_with_wight_decay(name,shape,stddev,wd):
    """
    生成所需要的变量，第二层函数，最底层函数用于便于修改gpu和cpu，我猜的
    :param name:
    :param shape:
    :param initializer:
    :return:

    """
    initializer = tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = "weight_loss")
        tf.add_to_collection('losses',weight_decay)
    return var
def interface(input_str):
    batch_size = 100
    word_vec_size = 60
    with tf.name_scope("conv1") as scope:

        kernel = _variable_with_wight_decay(
            name='weight',
            shape=[KERNEL_WIDTH, VEC_SIZE, 1, 64],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(input_str, kernel, [1, 1, 1, 1], padding="VALID")
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        _activation_summary(conv1)
    # pooling layer 1

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')
    # norm1
    norm1 = tf.nn.l2_normalize(pool1, 2)

    #drop out layer
    with tf.name_scope("dropout") as scope:
        dim = norm1.get_shape()
        rv = tf.random_uniform(dim,minval=PROPORTION,maxval=1+PROPORTION)

        mask = tf.floor(rv)
        dropout = norm1*mask
    # hidden layer 1
    with tf.name_scope("local3") as scope:
        reshape = tf.reshape(dropout,[batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_wight_decay('weights',shape=[dim,LOCAL_3],stddev=0.04,wd = 0.004)
        biases = _variable_on_cpu('biases',[LOCAL_3],tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(local3)
    # hidden layer 2
    with tf.name_scope("local4") as scope:
        weights = _variable_with_wight_decay('weights',shape=[LOCAL_3,LOCAL_4],stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',shape=[LOCAL_4],initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name = scope.name)
        _activation_summary(local4)
    #softmax生成层，按照参考的说法，tensorflow本身的交叉熵函数可以接受"not scaled"的数据，为了提高效率
    #softmax 对于原数据进行了线性变化

    with tf.name_scope('softmax_linear') as scope:
        weights = _variable_with_wight_decay('weights',shape=[LOCAL_4,CLASS_NUM],stddev= 1/192.0,wd=0.004)
        biases = _variable_on_cpu('biases',shape=[CLASS_NUM],initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name='softmax')

    return  softmax_linear
def loss(logits,label):
    """

    :param logits: 模型预测产生的结果
    :param label: 数据实际的值
    :return:

    """
    labels = tf.cast(label,tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total loss')

def _add_loss_summaries(total_loss):


  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(totalloss,global_step):
    """
     根据生成的损失训练模型

    :param totalloss:
    :param global_step:
    :return:
    """
    loss_average = _add_loss_summaries(total_loss=totalloss)
    lr = tf.train.exponential_decay(LEARNING_RATE,global_step,DECAY_STEP,DECAY_RATE)
    with tf.control_dependencies([loss_average]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grad = opt.compute_gradients(totalloss)

    apply = opt.apply_gradients(grads_and_vars=grad,global_step=global_step)
    for var in tf.trainable_variables():
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)
    #计算ema平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step
    )
    va_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply,va_op]):
        train = tf.no_op(name='train')
    return train
