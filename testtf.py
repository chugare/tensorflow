import tensorflow as tf


norm = tf.truncated_normal([2,5,5])

var = tf.Variable(tf.convert_to_tensor(norm))

mask = tf.random_uniform([2,5,5],minval=0.00,maxval=1.99)
# group = tf.group()
# var = tf.Variable()
init = tf.global_variables_initializer()
vp = tf.floor(norm)
get = vp*var
#pool = tf.nn.max_pool(t,[1,2,2,1],[1,2,2,1],'SAME')
reshape = tf.reshape(var,[5,-1])
var2 = tf.Variable(tf.zeros([5,6]))
with tf.Session() as session:
    result = session.run(init)
    s = var.eval(session)
    print(s)
    vk = session.run(get)
    print(vk)
