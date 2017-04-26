# coding:utf-8
import tensorflow as tf
class Interface( ):
    VEC_SIZE = 60
    KERNEL_WIDTH = 3
    KERNEL_WIDTH2 = 5
    KERNEL_WIDTH3 = 7
    PROPORTION = 1.0
    BATCH_SIZE = 100
    CAPCITY = 100000
    MIN_AFTER_QUEUE = 5000
    CONV_OUT = 128
    LOCAL_3 = 300
    LOCAL_4 = 200
    CLASS_NUM = 2
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY_FACTOR = 0.1
    DECAY_STEP = 1000
    DECAY_RATE = 0.98
    MOVING_AVERAGE_DECAY= 0.99
    FLAGS = tf.app.flags.FLAGS

    def custom_args(self,model):
        self.VEC_SIZE = 60
        self.CONV_OUT = model.num_of_kernel
        self.BATCH_SIZE = model.batch_size
        self.LOCAL_3 = model.local_1
        self.LOCAL_4 = model.local_2
        self.LEARNING_RATE = model.learning_rate
        self.MOVING_AVERAGE_DECAY = model.ema
        tf.app.flags.DEFINE_integer('batch_size', self.BATCH_SIZE,
                                    """Number of images to process in a batch.""")

    def _activation_summary(self,x):
        """
        为定义的操作添加总结，加入方便生成模型和流程图
        :param x:
        :return:
        """
        tensorname = x.op.name
        tf.summary.histogram(tensorname+"/activation",x)
        tf.summary.scalar(tensorname+'/sparsity',tf.nn.zero_fraction(x))
    def _variable_on_cpu(self,name,shape,initializer):
        """
        从示例代码里看到的函数方法，目的大概是创建缓存在cpu中的变量
        :param name: 变量名
        :param shape: 变量形状，大小维度
        :param initializer: 变量初始化工具，如随机初始化的truncate_normal
        :return: var 生成变量

        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer = initializer)
        return var
    def _variable_on_gpu(self,name,shape,initializer):
        """
        从示例代码里看到的函数方法，目的大概是创建缓存在gpu中的变量
        :param name: 变量名
        :param shape: 变量形状，大小维度
        :param initializer: 变量初始化工具，如随机初始化的truncate_normal
        :return: var 生成变量

        """
        with tf.device('/gpu:0'):
            var = tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer = initializer)
        return var
    def _variable_with_wight_decay(self,name,shape,stddev,wd):
        """
        生成所需要的变量，第二层函数，最底层函数用于便于修改gpu和cpu，我猜的
        :param name:
        :param shape:
        :param initializer:
        :return:

        """
        initializer = tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32)
        var = self._variable_on_cpu(name, shape, initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = "weight_loss")
            tf.add_to_collection('losses',weight_decay)
        return var
    def input (self,logits,labels):
        num_of_threads = 4
        logits_ , labels_ = tf.train.shuffle_batch([logits,labels],self.BATCH_SIZE,self.CAPCITY,self.MIN_AFTER_QUEUE,num_of_threads)
        return logits_, tf.reshape(labels_, [self.BATCH_SIZE])
    def interface(self,logits):

        with tf.variable_scope("conv3") as scope:

            kernel = self._variable_with_wight_decay(
                name='weight',
                shape=[self.KERNEL_WIDTH, self.VEC_SIZE, 1, self.CONV_OUT],
                stddev=5e-2,
                wd=0.0
            )
            conv = tf.nn.conv2d(logits, kernel, [1, 1, 1, 1], padding="VALID")
            biases = self._variable_on_cpu('biases1', [self.CONV_OUT], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name = scope.name)
            self._activation_summary(conv1)
        with tf.variable_scope("conv5") as scope:
            kernel = self._variable_with_wight_decay(
                name='weight',
                shape=[self.KERNEL_WIDTH2, self.VEC_SIZE, 1, self.CONV_OUT],
                stddev=5e-2,
                wd=0.0
            )
            conv2 = tf.nn.conv2d(logits, kernel, [1, 1, 1, 1], padding="VALID")
            biases2 = self._variable_on_cpu('biases2', [self.CONV_OUT], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv2, biases2)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(conv2)
        with tf.variable_scope("conv7") as scope:
            kernel = self._variable_with_wight_decay(
                name='weight',
                shape=[self.KERNEL_WIDTH3, self.VEC_SIZE, 1, self.CONV_OUT],
                stddev=5e-2,
                wd=0.0
            )
            conv3 = tf.nn.conv2d(logits, kernel, [1, 1, 1, 1], padding="VALID")
            biases3 = self._variable_on_cpu('biases3', [self.CONV_OUT], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv3, biases3)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            self. _activation_summary(conv3)

        # pooling layer 1
        length = conv1.get_shape()

        pool1 = tf.nn.max_pool(conv1, ksize=[1, length[1].value , length[2].value, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')
        # norm1
        norm1 = tf.nn.l2_normalize(pool1, 3)

        length = conv2.get_shape()
        pool2 = tf.nn.max_pool(conv2, ksize=[1, length[1].value , length[2].value, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')
        # norm1
        norm2 = tf.nn.l2_normalize(pool2, 3)

        length = conv3.get_shape()
        pool3= tf.nn.max_pool(conv3, ksize=[1, length[1].value , length[2].value, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')
        # norm1
        norm3 = tf.nn.l2_normalize(pool3, 3)

        norm = tf.concat([norm1,norm2,norm3],2)
        #drop out layer
        with tf.variable_scope("dropout") as scope:
            dim = norm.get_shape()
            rv = tf.random_uniform(dim,minval=self.PROPORTION,maxval=1+self.PROPORTION)

            mask = tf.floor(rv)
            dropout = norm*mask
        # hidden layer 1
        with tf.variable_scope("local3") as scope:
            dim = dropout.get_shape()[0].value
            dropout = tf.reshape(dropout,[dim,-1])
            dim2 = dropout.get_shape()[1].value
            weights = self._variable_with_wight_decay('weights',shape=[dim2,self.LOCAL_3],stddev=0.04,wd = 0.00)
            biases = self._variable_on_cpu('biases',[self.LOCAL_3],tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(dropout,weights)+biases,name=scope.name)
            self._activation_summary(local3)
        # hidden layer 2
        with tf.variable_scope("local4") as scope:
            weights = self._variable_with_wight_decay('weights',shape=[self.LOCAL_3,self.LOCAL_4],stddev=0.04,wd=0.001)
            biases = self._variable_on_cpu('biases',shape=[self.LOCAL_4],initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name = scope.name)
            self._activation_summary(local4)
        # softmax生成层，按照参考的说法，tensorflow本身的交叉熵函数可以接受"not scaled"的数据，为了提高效率
        # softmax 对于原数据进行了线性变化

        with tf.variable_scope('softmax_linear') as scope:
            dim = local4.get_shape()
            weights = self._variable_with_wight_decay('weights',shape=[dim[1], self.CLASS_NUM],stddev= 1/192.0,wd=0.002)
            biases = self._variable_on_cpu('biases',shape=[self.CLASS_NUM],initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4,weights),biases,name='softmax')

        return  softmax_linear
    def loss(self,logits,label):
        """

        :param logits: 模型预测产生的结果
        :param label: 数据实际的值
        :return:

        """
        labels = tf.cast(label, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
        tf.add_to_collection('losses',cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'),name='total_loss')

    def _add_loss_summaries(self,total_loss):


      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      losses = tf.get_collection('losses')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

      for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

      return loss_averages_op

    def train(self,totalloss,global_step):
        """
         根据生成的损失训练模型

        :param totalloss:
        :param global_step:
        :return:
        """
        loss_average = self._add_loss_summaries(total_loss=totalloss)
        lr = tf.train.exponential_decay(self.LEARNING_RATE, global_step, self.DECAY_STEP, self.LEARNING_RATE_DECAY_FACTOR)
        with tf.control_dependencies([loss_average]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(totalloss)

        apply = opt.apply_gradients(grads_and_vars=grads,global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        # 计算ema平均
        variable_averages = tf.train.ExponentialMovingAverage(
            self.MOVING_AVERAGE_DECAY,
            global_step
        )
        va_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([apply,va_op]):
            train = tf.no_op(name='train')
        return train
