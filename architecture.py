import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Mobilenetv2(nn.Module):
    def __init__(self):
        super(Mobilenetv2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=self.model.classifier[1].in_features, out_features=128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )   
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.model.classifier(x)
        return x