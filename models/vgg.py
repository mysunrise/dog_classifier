import torch.nn as nn


class Vgg19Ft(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(Vgg19Ft, self).__init__()
        self.features = pretrained_model.features
        mod = list(pretrained_model.classifier.children())[:-1]
        mod.append(nn.Linear(4096, num_classes))
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
