import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

tf.logging.set_verbosity(old_v)

n = 10
batch_size = 100

x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])


def model(data):
    w=tf.Variable(tf.random_normal([128, 10],stddev=0.1))
    b=tf.Variable(tf.random_normal([10]))
    data=tf.transpose(data,[1,0,2])
    data=tf.reshape(data,[-1,28])
    data=tf.split(data,28,0)
    lstm_cell=rnn.BasicLSTMCell(128)
    outputs, states=rnn.static_rnn(lstm_cell,data,dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1],w),b)

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
                x1=x1.reshape((batch_size,28,28))

                _, c = sess.run([optimize, cost], feed_dict={x: x1, y: y1})
                loss += c
            print(loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images.reshape((-1,28,28)), y: mnist.test.labels}))


train(x)