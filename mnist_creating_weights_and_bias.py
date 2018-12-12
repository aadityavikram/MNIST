import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

tf.logging.set_verbosity(old_v)

nnh1 = 500
nnh2 = 500
nnh3 = 500
n = 10
batch_size = 100
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


def model(data):
    w1=tf.Variable(tf.random_normal([784, nnh1],stddev=0.1))
    b1=tf.Variable(tf.zeros([nnh1]))

    w2=tf.Variable(tf.random_normal([nnh1, nnh2],stddev=0.1))
    b2=tf.Variable(tf.zeros([nnh2]))

    w3=tf.Variable(tf.random_normal([nnh2, nnh3],stddev=0.1))
    b3=tf.Variable(tf.zeros([nnh3]))

    wo=tf.Variable(tf.zeros([nnh3, n]))
    bo=tf.Variable(tf.zeros([n]))

    l1 = tf.add(tf.matmul(data,w1),b1)
    l1 = tf.sigmoid(l1)
    l2 = tf.add(tf.matmul(l1,w2),b2)
    l2 = tf.sigmoid(l2)
    l3 = tf.add(tf.matmul(l2,w3),b3)
    l3 = tf.sigmoid(l3)
    output = tf.add(tf.matmul(l3,wo),bo)

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