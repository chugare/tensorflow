import tensorflow as tf
import numpy

VEC_SIZE = 60
KERNEL_WIDTH = 3

def _veriable_on_cpu(name,shape,initializer):
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
def _veriable_with_wight_decay(name,shape,stddev,wd):
    """
    生成所需要的变量，第二层函数，最底层函数用于便于修改gpu和cpu，我猜的
    :param name:
    :param shape:
    :param initializer:
    :return:

    """
    initializer = tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32)
    var = _veriable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = "weight_loss")
        tf.add_to_collection('losses',weight_decay)
    return var
def interface(input_str):
    batch_size = 100
    word_vec_size = 60
    with tf.name_scope("cov1") as scope:

        kernel = _veriable_with_wight_decay(
            name='weight',
            shape=[KERNEL_WIDTH, VEC_SIZE, 1, 64],
            stddev=5e-2,
            wd=0.0
        )
        conv = tf.nn.conv2d(input_str, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _veriable_on_cpu('biases', [64], tf.constant_initializer(0.0))


