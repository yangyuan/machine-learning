import datetime
import tensorflow as tf
import numpy as np


class BaseModel:
    def __init__(self, _name):
        self.name = _name
        self.parameters = dict()
        self.specification = dict()


class MinBatch:
    def __init__(self, tensors, batch_size):
        self.n = tensors[0].shape[0]
        self.batch_size = batch_size
        self.tensors = tensors
        self.indexes = np.arange(self.n)
        np.random.shuffle(self.indexes)
        self.offset = 0

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def __next__(self):
        if self.offset < self.n:
            offset = self.offset
            self.offset += self.batch_size
            max_offset = min(self.n, self.offset)
            indexes = self.indexes[offset:max_offset]
            return (x[indexes] for x in self.tensors)
        else:
            raise StopIteration()


class BaseData:
    def __init__(self, _name):
        self.name = _name

        self.training_x = None
        self.training_y = None

        self.validation_x = None
        self.validation_y = None

        self.test_x = None
        self.test_y = None

    def get_training(self):
        return tf.data.Dataset.from_tensor_slices((self.training_x, self.training_y))

    def validate_shape(self, _shape_x, _shape_y):
        if len(_shape_x) != len(self.training_x.shape)\
                or len(_shape_y) != len(self.training_y.shape):
            return False

        for expected, actual in zip(_shape_x, self.training_x.shape):
            if expected is None:
                continue
            elif expected != actual:
                return False

        return True


class OneHotData(BaseData):
    def __init__(self, _name, _num_labels,
                 _training_x, _training_y,
                 _validation_x, _validation_y,
                 _test_x, _test_y):
        super().__init__(_name)

        assert (len(_training_y.shape) == 2)
        m, n = _training_y.shape
        assert (n == _num_labels)
        assert (_training_x.shape[0] == m)

        self.training_x = _training_x
        self.training_y = _training_y

        self.validation_x = _validation_x
        self.validation_y = _validation_y

        self.test_x = _test_x
        self.test_y = _test_y


class Logger:
    @staticmethod
    def log(string):
        print('%s %s' % (datetime.datetime.now(), string))
