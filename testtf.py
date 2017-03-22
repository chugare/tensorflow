import tensorflow as tf



norm = tf.truncated_normal([2,5,5])
label = tf.ones([2])
var = tf.Variable(tf.convert_to_tensor(norm))

mask = tf.random_uniform([2,5,5],minval=0.00,maxval=1.99)

norm = tf.random_normal([2,5,5,1])

# group = tf.group()
var = tf.Variable(norm,name='var1')
init = tf.global_variables_initializer()
a = range(1,10)
def vec_input(k):
    for i in range(0,k):
        for j in range(0,k):
            yield i,j
    return
up = vec_input(10)
while up.next():
    print up.next()
vp = tf.floor(norm)
get = vp*var
pool = tf.nn.max_pool(norm,[1,2,2,1],[1,2,2,1],'SAME')
reshape = tf.reshape(var,[5,-1])
var2 = tf.Variable(tf.zeros([5,6]),name='var2')

pool = tf.nn.max_pool(var,[1,2,2,1],[1,2,2,1],'SAME')

