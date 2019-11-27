from torch import nn
from torchscope import scope
from torchvision import models


class FrameDetectionModel(nn.Module):
    def __init__(self):
        super(FrameDetectionModel, self).__init__()
        # model = models.resnet50(pretrained=True)
        model = models.mobilenet_v2(pretrained=True)
        # Remove linear layer (since we're not doing classification)
        modules = list(model.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc = nn.Linear(1280, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.features(images)  # [N, 2048, 1, 1]
        # x = x.view(-1, 2048)  # [N, 2048]
        # x = self.fc(x)
        # x = self.sigmoid(x)  # [N, 8]
        return x


if __name__ == "__main__":
    model = FrameDetectionModel()
    scope(model, (3, 224, 224))
