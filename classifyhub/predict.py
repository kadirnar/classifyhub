# Code written by Kadir Nar, 2023

from classifyhub.libhub.timm_models import TimmClassificationModel


class ClassifyPredictor:
    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        model = TimmClassificationModel(self.model_name)
        self.model = model
        return self.model

    def predict(self, img_path):
        return self.model.predict(img_path)


if __name__ == "__main__":
    model = ClassifyPredictor("resnet18")
    print(model.predict("data/plane.jpg"))
