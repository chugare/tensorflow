import tensorflow as tf


<<<<<<< Updated upstream
norm = tf.truncated_normal([2,5,5])

var = tf.Variable(tf.convert_to_tensor(norm))

mask = tf.random_uniform([2,5,5],minval=0.00,maxval=1.99)
=======
norm = tf.random_normal([2,5,5,1])
>>>>>>> Stashed changes
# group = tf.group()
var = tf.Variable(norm)
init = tf.global_variables_initializer()
<<<<<<< Updated upstream
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
=======
pool = tf.nn.max_pool(var,[1,2,2,1],[1,2,2,1],'SAME')
with tf.Session() as session:
    session.run(init)
    result = session.run(pool)
    print var.eval(session)
    print result
>>>>>>> Stashed changes
