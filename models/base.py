

class BaseModel:
    def __init__(self, _name):
        self.name = _name


class BaseData:
    def __init__(self, _name):
        self.name = _name

        self.training_x = None
        self.training_y = None

        self.validation_x = None
        self.validation_y = None

        self.test_x = None
        self.test_y = None


class OneHotData(BaseData):
    def __init__(self, _name,
                 _training_x, _training_y,
                 _validation_x, _validation_y,
                 _test_x, _test_y):
        super().__init__(_name)

        print(_training_x.shape)
        exit()
