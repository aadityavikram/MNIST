import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

tf.logging.set_verbosity(old_v)

n = 10
batch_size = 128
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def model(data):
    W_conv1=tf.Variable(tf.random_normal([5,5,1,4]))
    b_conv1=tf.Variable(tf.zeros([4]))

    W_conv2=tf.Variable(tf.random_normal([3,3,4,8]))
    b_conv2=tf.Variable(tf.zeros([8]))

    W_conv3 = tf.Variable(tf.random_normal([3, 3, 8, 16]))
    b_conv3 = tf.Variable(tf.zeros([16]))

    W_fc=tf.Variable(tf.random_normal([7*7*16,256]))
    b_fc=tf.Variable(tf.zeros([256]))

    W_out=tf.Variable(tf.zeros([256, n]))
    b_out=tf.Variable(tf.zeros([n]))

    X=tf.reshape(x, shape=[-1,28,28,1])

    conv1=tf.sigmoid(tf.nn.conv2d(X,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)

    conv2=tf.sigmoid(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2)
    conv2=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.sigmoid(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc=tf.reshape(conv3,[-1,7*7*16])
    fc=tf.sigmoid(tf.matmul(fc,W_fc)+b_fc)

    output=tf.matmul(fc,W_out)+b_out

    return output


def train(data):
    prediction = model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimize = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x1, y1 = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimize, cost], feed_dict={x: x1, y: y1})
                loss += c
            print(loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train(x)