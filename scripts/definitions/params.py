from scripts.utils.constants import C
from scripts.utils.serialization import to_json_dict, from_json_dict, defaults


class TransferModelParams:
    """
    Class representing transfer model configuration
    """

    def __init__(self, name, output_size, views_path):
        self.name = name
        # size of model's output -> this will be input size for our network
        self.output_size = output_size
        # directory with Views for this model
        self.views_path = views_path


T_MODELS_PARAMS = [
    TransferModelParams("alexm2", 4096, C.ALEX_NET_M2_PATH),
    TransferModelParams("alex", 1000, C.ALEX_NET_PATH),
    TransferModelParams("resnet", 2048, C.RES_NET_PATH),
    TransferModelParams("vggm2", 4096, C.VGG_M2_PATH),
    TransferModelParams("vgg", 25088, C.VGG_PATH)
]


def get_t_model_params(name):
    return [m for m in T_MODELS_PARAMS if m.name == name][0]


class RunParams:
    """
    Aggregation of all parameters for a training run.
    """

    def __init__(self):
        # name of the test - used to identify it later
        self.name = None
        self.output_dir = None

        # list of our models to be tested
        self.models = None
        # list of transfer learning models to be tested
        self.t_models = None

        # neural network optimization method
        self.optimizer = None
        # number of iterations to be run
        self.iterations = None
        self.dataset_params: DatasetParams = None
        self.train_params: TrainParams = None

    def to_json_dict(self):
        """
        Serializes itself to a Python dictionary that can be later serialized to json using default json serialization
        config.
        """
        return to_json_dict(self, {
            "dataset_params": self.dataset_params.to_json_dict(),
            "train_params": self.train_params.to_json_dict(),
        })

    @staticmethod
    def from_json_dict(dic):
        """
        Deserializes itself from Python dictionary.

        :param dic: Python dictionary having all values saved during serialization
        """
        params = from_json_dict(dic, RunParams, {
            "dataset_params": lambda val: DatasetParams.from_json_dict(val),
            "train_params": lambda val: TrainParams.from_json_dict(val)
        })
        return defaults(params, lambda p: {
            "output_dir": p.name
        })

    @staticmethod
    def empty():
        return RunParams()


class Combination:
    """
    Represents a combination transfer learning model and architecture.
    """

    def __init__(self, model, t_model):
        # name of our model (Feature Extractor + Classifier)
        self.model = model
        # name of pretrained CNN network whose output should be used as input for our model
        self.t_model = t_model

    def equals(self, comb):
        return self.model == comb.model and self.t_model == comb.t_model

    def to_json_dict(self):
        return str((self.model, self.t_model))

    @staticmethod
    def from_json_dict(dic):
        (model, t_model) = eval(dic)
        return Combination(model, t_model)

    @staticmethod
    def empty():
        return Combination(None, None)


class DatasetParams:
    """
    Parameters regarding dataset preparation.
    """

    def __init__(self):
        # file containing already prepared Samples
        self.samples_file_name = None
        # file describing how to split samples into train, dev & test sets
        self.samples_split = None
        # weights for labels {0,1} during loss calculation
        self.labels_weights = None

    def to_json_dict(self):
        return to_json_dict(self)

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, DatasetParams)

    @staticmethod
    def empty():
        return DatasetParams()


class TrainParams:
    """
    Parameters regarding training
    """

    def __init__(self):
        self.batch_size = None
        # number of epochs to run
        self.epochs = None

    def to_json_dict(self):
        return to_json_dict(self)

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, TrainParams)

    @staticmethod
    def empty():
        return TrainParams()
