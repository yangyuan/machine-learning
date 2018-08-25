import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from .base import BaseModel


class LeNet5(BaseModel):
    def __init__(self, _num_labels=10, initializer_mean=0, initializer_stddev=0.3, learning_rate=0.001):
        super().__init__('LeNet-5')

        self.parameters['mean'] = initializer_mean
        self.parameters['stddev'] = initializer_stddev

        self.specification['shape_input'] = (32, 32, 1)
        self.specification['shape_output'] = (_num_labels,)

        x = tf.placeholder(tf.float32, shape=(None,) + self.specification['shape_input'])
        self.y = tf.placeholder(tf.int32, shape=(None,) + self.specification['shape_output'])

        # C1
        # In: (32, 32, 1), Out: (28, 28, 6)
        c1_kernel = tf.Variable(
            tf.truncated_normal(
                shape=[5, 5, 1, 6],
                mean=self.parameters['mean'], stddev=self.parameters['std']))
        c1_bias = tf.get_variable(
            shape=[6],
            initializer=tf.random_normal_initializer(stddev=self.parameters['std']))
        c1 = tf.nn.conv2d(x, c1_kernel, [1, 1, 1, 1], padding='VALID') + c1_bias

        # S1
        # In: (28, 28, 6), Out: (14, 14, 6)
        s2 = tf.nn.avg_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        s2_coefficient = tf.get_variable(
            shape=[6],
            initializer=tf.random_normal_initializer(stddev=self.parameters['std']))
        s2_bias = tf.get_variable(
            shape=[6],
            initializer=tf.random_normal_initializer(stddev=self.parameters['std']))
        s2 = tf.nn.tanh(s2 * s2_coefficient + s2_bias)

        # Layer 2: Input 14x14x6, Output 10x10x16
        self.conv2_kernels = tf.Variable(
            tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.parameters['mean'], stddev=self.parameters['std']))
        self.conv2_biases = tf.get_variable(
            shape=[16],
            initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_kernels, [1, 1, 1, 1], padding='VALID') + self.conv2_biases
        # Pooling -> from 10x10x16 to 5x5x16
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Activation 2
        self.conv2 = tf.nn.relu(self.pool2)

        # Flatten -> from 5x5x16 to 400x1
        self.flattened = flatten(self.conv2)

        # Fully Connected Layer n.1
        self.fcl1_weights = tf.Variable(
            tf.truncated_normal(
                shape=[400, 120],
                mean=self.parameters['mean'],
                stddev=self.parameters['stddev']))
        self.fcl1_biases = tf.get_variable(
            shape=[120],
            initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        self.fcl1 = tf.matmul(self.flattened, self.fcl1_weights) + self.fcl1_biases
        # Activation 3
        self.fcl1 = tf.nn.relu(self.fcl1)

        # Fully Connected Layer n.2
        self.fcl2_weights = tf.Variable(
            tf.truncated_normal(shape=[120, 84], mean=self.parameters['mean'], stddev=self.parameters['stddev']))
        self.fcl2_biases = tf.get_variable(name="fc2_biases", shape=[84],
                                           initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        self.fcl2 = tf.matmul(self.fcl1, self.fcl2_weights) + self.fcl2_biases
        # Activation 4
        self.fcl2 = tf.nn.relu(self.fcl2)

        # Fully Connected Layer n.3
        self.fcl3_weights = tf.Variable(
            tf.truncated_normal(shape=[84, 10], mean=self.parameters['mean'], stddev=self.parameters['stddev']))
        self.fcl3_biases = tf.get_variable(
            shape=[10],
            initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        self.logits = tf.matmul(self.fcl2, self.fcl3_weights) + self.fcl3_biases

        # Loss and metrics
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss_op = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_step = self.optimizer.minimize(self.loss_op)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def train(self, epochs, batch_size, auto_save=True):
        assert (epochs > 0 and batch_size > 0)

        num_examples = len(self.train_data)

        print('Training the model . . .')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    X_batch, y_batch = self.train_data[offset:end], self.train_labels[offset:end]

                    _, acc, cross = session.run([self.training_step, self.accuracy_operation, self.cross_entropy],
                                                feed_dict={self.X: X_batch, self.y: y_batch})

                if self.validation_data is not None:
                    validation_accuracy = self.evaluate(self.validation_data, self.validation_labels, batch_size)
                    print("Epoch {} - validation accuracy {:.3f} ".format(epoch + 1, validation_accuracy))

                if auto_save and (epoch % 10 == 0):
                    save_path = self.saver.save(session, 'tmp/model.ckpt'.format(epoch))

            test_accuracy = self.evaluate(self.test_data, self.test_labels, batch_size=batch_size)
            return test_accuracy

    def evaluate(self, X_data, y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def restore_model(self, path):
        with tf.Session() as session:
            self.saver.restore(sess=session, save_path=path)