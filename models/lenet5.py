import tensorflow as tf
from .base import BaseModel, Logger


class LeNet5(BaseModel):
    def __init__(self, _num_labels=10, initializer_mean=0, initializer_stddev=0.3, learning_rate=0.001, original=False):
        super().__init__('LeNet-5')

        self.parameters['mean'] = initializer_mean
        self.parameters['stddev'] = initializer_stddev

        self.specification['shape_input'] = (32, 32, 1)
        self.specification['shape_output'] = (_num_labels,)

        self.x = tf.placeholder(tf.float32, shape=(None,) + self.specification['shape_input'])
        self.y = tf.placeholder(tf.int32, shape=(None,) + self.specification['shape_output'])

        # C1
        # In: (32, 32, 1), Out: (28, 28, 6)
        c1_kernel = tf.get_variable(
            'c1_kernel',
            shape=[5, 5, 1, 6],
            initializer=tf.truncated_normal_initializer(stddev=self.parameters['stddev']))
        c1_bias = tf.get_variable(
            'c1_bias',
            shape=[6],
            initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        c1 = tf.nn.conv2d(self.x, c1_kernel, [1, 1, 1, 1], padding='VALID') + c1_bias

        # S2
        # In: (28, 28, 6), Out: (14, 14, 6)
        if original:
            s2 = tf.nn.avg_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            s2_coefficient = tf.get_variable(
                's2_coefficient',
                shape=[6],
                initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
            s2_bias = tf.get_variable(
                's2_bias',
                shape=[6],
                initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
            s2 = tf.nn.tanh(s2 * s2_coefficient + s2_bias)
        else:
            s2 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            s2 = tf.nn.relu(s2)

        # C3
        # In: (14, 14, 6), Out: (10, 10, 16)
        c3_kernel = tf.get_variable(
            'c3_kernel',
            shape=[5, 5, 6, 16],
            initializer=tf.truncated_normal_initializer(stddev=self.parameters['stddev']))
        c3_bias = tf.get_variable(
            'c3_bias',
            shape=[16],
            initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
        c3 = tf.nn.conv2d(s2, c3_kernel, [1, 1, 1, 1], padding='VALID') + c3_bias

        # S4
        # In: (10, 10, 16), Out: (5, 5, 16)
        if original:
            s4 = tf.nn.avg_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            s4_coefficient = tf.get_variable(
                's4_coefficient',
                shape=[16],
                initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
            s4_bias = tf.get_variable(
                's4_bias',
                shape=[16],
                initializer=tf.random_normal_initializer(stddev=self.parameters['stddev']))
            s4 = tf.nn.tanh(s4 * s4_coefficient + s4_bias)
        else:
            s4 = tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            s4 = tf.nn.relu(s4)

        # C5
        # In: (5, 5, 16), Out: (120, )
        flattened = tf.contrib.layers.flatten(s4)  # (5, 5, 16) to (400, )
        c5 = tf.contrib.layers.fully_connected(
            flattened,
            120,
            activation_fn=tf.nn.tanh if original else tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=self.parameters['stddev']),
            weights_regularizer=None,
            biases_initializer=tf.random_normal_initializer(stddev=self.parameters['stddev'])
        )

        # C6
        # In: (120, ), Out: (84, )
        c6 = tf.contrib.layers.fully_connected(
            c5,
            84,
            activation_fn=tf.nn.tanh if original else tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=self.parameters['stddev']),
            weights_regularizer=None,
            biases_initializer=tf.random_normal_initializer(stddev=self.parameters['stddev'])
        )

        # OUTPUT
        # In: (84, ), Out: ({_num_labels}, )
        self.logits = tf.contrib.layers.fully_connected(
            c6,
            _num_labels,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=self.parameters['stddev']),
            weights_regularizer=None,
            biases_initializer=tf.random_normal_initializer(stddev=self.parameters['stddev'])
        )

        # Loss and metrics
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss_op = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_step = self.optimizer.minimize(self.loss_op)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def train(self, data, epochs, batch_size, auto_save=True):
        assert (epochs > 0 and batch_size > 0)

        num_examples = len(data.training_x)

        Logger.log('Training the model . . .')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):

                train_data, train_labels = data.training_x, data.training_y
                xxxxxxxx = int(num_examples/batch_size)
                i = 0
                for offset in range(0, num_examples, batch_size):
                    i += 1
                    end = offset + batch_size
                    x_batch, y_batch = train_data[offset:end], train_labels[offset:end]

                    _, acc, cross = session.run([self.training_step, self.accuracy_operation, self.cross_entropy],
                                                feed_dict={self.x: x_batch, self.y: y_batch})
                    print(i, xxxxxxxx, acc)

                """
                dataset = data.get_training().shuffle(batch_size*16).repeat().batch(batch_size)
                iterator = dataset.make_one_shot_iterator()
                next_element = iterator.get_next()

                xxxxxxxx = int(num_examples/batch_size)

                for i in range(int(num_examples/batch_size)):
                    _x, _y = session.run(next_element)
                    _, acc, cross = session.run([self.training_step, self.accuracy_operation, self.cross_entropy],
                                                feed_dict={self.x: _x, self.y: _y})
                    print(i, xxxxxxxx, acc)
                    
                    
                """

                validation_accuracy = self.evaluate(data.validation_x, data.validation_y, batch_size)
                Logger.log("Epoch {} - validation accuracy {:.3f} ".format(epoch + 1, validation_accuracy))

                if auto_save and (epoch % 10 == 0):
                    save_path = self.saver.save(session, 'data/checkpoints/model.ckpt'.format(epoch))

            test_accuracy = self.evaluate(self.test_data, self.test_labels, batch_size=batch_size)
            return test_accuracy

    def evaluate(self, X_data, y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def restore_model(self, path):
        with tf.Session() as session:
            self.saver.restore(sess=session, save_path=path)