import os

import torch

from config import config
from logger_conf import setup_logger
from yolo.yolov5.models.common import DetectMultiBackend
from yolo._hdfs import HDFSUtils

logger = setup_logger("model_rep")


class ModelParent(object):
    def __init__(self, model_key: str, weight_type, weight_path: str):
        self.weight_type = weight_type
        self.model = None
        self.model_key = model_key
        self.weight_path = weight_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model(self):
        self.model = DetectMultiBackend(self.download_model(), device=self.device)
        logger.info(f'进程{os.getpid()}从【{self.model_key}】加载模型【{self.weight_path}】加载成功')

    def get_model(self):
        return self.model

    def predict(self, image):
        pass

    def download_model(self):
        if self.weight_type == 'hdfs':
            return HDFSUtils.download(self.weight_path, "/home/temp/model")

        else:
            return self.weight_path


class PersonModel(ModelParent):
    model_key = 'person_model'

    def __init__(self, model_key=model_key, weight_type='hdfs', weight_path=config['model_weights'][model_key]):
        super(PersonModel, self).__init__(model_key, weight_type, weight_path)

    def predict(self, image):
        return self.model(image)


class HelmetModel(ModelParent):
    model_key = 'helmet_model'

    def __init__(self, model_key=model_key, weight_type='hdfs', weight_path=config['model_weights'][model_key]):
        super(HelmetModel, self).__init__(model_key, weight_type, weight_path)

    def predict(self, image):
        return self.model(image)


class ModelRep(object):
    _instances = {}

    @classmethod
    def load_models(cls):
        person_model = PersonModel()
        person_model.init_model()
        helmet_model = HelmetModel()
        helmet_model.init_model()
        cls.register_model(person_model.model_key, person_model)
        cls.register_model(helmet_model.model_key, helmet_model)

    @classmethod
    def register_model(cls, model_key: str, model: ModelParent):
        if model_key in cls._instances.keys():
            return
        cls._instances[model_key] = model

    @classmethod
    def contains_model(cls, model_key: str):
        return model_key in cls._instances.keys()

    @classmethod
    def get_model(cls, model_key):
        model = cls._instances[model_key]
        if model is None:
            raise ValueError(f"模型 '{model_key}'不存在.")
        return model
