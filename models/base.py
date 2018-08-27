

class BaseModel:
    def __init__(self, _name):
        self.name = _name
        self.parameters = dict()
        self.specification = dict()


class BaseData:
    def __init__(self, _name):
        self.name = _name

        self.training_x = None
        self.training_y = None

        self.validation_x = None
        self.validation_y = None

        self.test_x = None
        self.test_y = None

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

