import tensorflow as tf

a= []
b = []
for i in range(0,100):
    a.append([[[1]]])
    b.append(1)


norm = tf.convert_to_tensor(a,tf.float32)
label = tf.convert_to_tensor(b,tf.float32)
norms = []
labels = []
kernel = tf.Variable(tf.ones([2, 2, 1, 1]))
nb = tf.ones([1, 5, 5, 1],name='input')
var = tf.Variable(tf.convert_to_tensor(norm))

mask = tf.random_uniform([2,5,5],minval=0.00,maxval=1.99)


# group = tf.group()
init = tf.global_variables_initializer()
a = range(1,10)
vp = tf.floor(norm)
get = vp*var
var2 = tf.Variable(tf.zeros([5,6]),name='var2')
conv = tf.nn.conv2d(nb, kernel, [1,1,1,1], 'VALID')
dim = nb.get_shape()
pool = tf.nn.max_pool(nb,[1,,2,1],[1,2,2,1],'VALID')

with tf.Session() as ss:
    ss.run(init)
    print (kernel.eval(ss))
    result = ss.run(pool)
    print (result)