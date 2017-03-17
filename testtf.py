import tensorflow as tf

ones = tf.ones([2,2,2])

norm = tf.random_normal([2,2,2])
tuple = tf.tuple([ones,norm])
# group = tf.group()
# var = tf.Variable()
init = tf.global_variables_initializer()
#pool = tf.nn.max_pool(t,[1,2,2,1],[1,2,2,1],'SAME')
with tf.Session() as session:
    result = session.run(tuple)

    print result