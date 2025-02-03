import torch
import torch.nn as nn
import torchvision


class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet18'):
        """
            parameters: a backbone network name for feature extractor, must be supported by torchvision.models

                backbone: a backbone network name
        """
        super.__init__()

        if backbone not in dir(torchvision.models):
            raise (f"{backbone} is not a supported backbone model")

        self.feature_extractor = getattr(
            torchvision.models, backbone)(pretrained=True)
        self.feature_extractor.modules
        out_features = list(self.feature_extractor.modules())[-1].out_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """ 
            parameters: pairs of two images

                img1: tensor with shape=[batch * 3 * 224 * 224]
                img2: tensor with shape=[batch * 3 * 224 * 224]

            returns: similarity of each pair of images

                output: tensor with shape=[batch * 1]

        """
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)

        combined_features = feat1 * feat2

        output = self.classifier(combined_features)
        return output
