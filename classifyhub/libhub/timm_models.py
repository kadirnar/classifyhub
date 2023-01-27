# Code written by Kadir Nar, 2023

import urllib

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmClassificationModel:
    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name
        self.load()

    def load_model(self):
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.model.eval()

    def load_classes(self):
        url, filename = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "imagenet_classes.txt",
        )
        urllib.request.urlretrieve(url, filename)
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def load_transform(self):
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)

    def load(self):
        self.load_model()
        self.load_transform()
        self.load_classes()

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        return [(self.categories[top5_catid[i]], top5_prob[i].item()) for i in range(top5_prob.size(0))]
