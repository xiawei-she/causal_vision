import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights,ViT_L_16_Weights
from models import vit


class Backbone(nn.Module):
    def __init__(self,backbone_type,backbone_only=True):
        super(Backbone, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.backbone_only = backbone_only

        self.backbone_type = backbone_type
        assert backbone_type in {'cnn','vit'}, 'wrong backbone type'

        if self.backbone_type == 'cnn':
            self.model = models.resnet34(weights=ResNet34_Weights)
            if backbone_only:
                self.model.fc = nn.Linear(1000, 1)
        elif self.backbone_type == 'vit':
            self.model = models.vit_l_16(weights=ViT_L_16_Weights)
            if backbone_only:
                self.model.fc = nn.Linear(1000, 1)

    def forward(self, x):
        if self.backbone_only:
            return self.model(x)
        else:
            return self.sigmoid(self.model(x))


if __name__ == '__main__':
    backbone = Backbone('vit')
    print(backbone)


