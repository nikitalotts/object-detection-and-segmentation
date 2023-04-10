from src.models.BaseModel import BaseModel
from ultralytics import YOLO


class YoloModel(BaseModel):
    def __init__(self, model_path: str, task: str):
        print(model_path)
        self.estimator = YOLO(model_path, task=task)

    def train(self, data: str, imgsz: int, epochs: int, batch: int):
        self.estimator.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch)

    def val(self, data: str, imgsz: int):
        results = self.estimator.val(data=data, imgsz=imgsz)
        output = {
            'mAP50': results.results_dict['metrics/mAP50(M)'],
            'precision': results.results_dict['metrics/precision(B)'],
            'recall': results.results_dict['metrics/recall(B)'],
            'f1': results.box.f1[0]
        }
        return output

    def predict(self, source: str, task: str, save: bool, save_txt: bool, stream: bool):
        generators = self.estimator.predict(source=source, task=task, save=save, save_txt=save_txt, stream=stream)
        for _ in generators:
            pass

