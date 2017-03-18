import tensorflow as tf

ones = tf.range(0,10,1,tf.float32)

var = tf.Variable(tf.convert_to_tensor([ones,ones,ones]))

norm = tf.random_normal([2,2,2])
# group = tf.group()
# var = tf.Variable()
init = tf.global_variables_initializer()
#pool = tf.nn.max_pool(t,[1,2,2,1],[1,2,2,1],'SAME')
reshape = tf.reshape(var,[5,-1])
var2 = tf.Variable(tf.zeros([5,6]))
with tf.Session() as session:
    result = session.run(init)
    s = var.eval(session)
    print(s)

    vk = session.run(reshape)
    s = var2.load(vk, session)
    ss = var2.eval(session)
    print(s)
    print(ss)
    print(vk)
