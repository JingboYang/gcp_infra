import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):

    def __init__(self, base_model, num_classes):
        super(EmbeddingModel, self).__init__()

        self.model = base_model

        if self.model.__class__.__name__.lower() == 'densenet':
            self.pool = nn.AdaptiveAvgPool2d(1)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif self.model.__class__.__name__.lower() == 'resnet':
            num_ftrs = self.model.fc.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)

    # Adopted from CheXpert
    def forward(self, x):
        

        if self.model.__class__.__name__.lower() == 'densenet':
            x = self.model.features(x)
            x = F.relu(x, inplace=True)
            emb = self.pool(x).view(x.size(0), -1)
            x = self.model.classifier(emb)
            return x, emb
        elif self.model.__class__.__name__.lower() == 'resnet':
            

            # for layer in self.model.modules():
            for layer in self.model.children():

                if layer.__class__.__name__.lower() == 'adaptiveavgpool2d':
                    break
                else:
                    x = layer(x)

            x = self.model.avgpool(x)
            x = F.relu(x, inplace=True)
            emb = x.view(x.size(0), -1)
            x = self.model.classifier(emb)
            return x, emb

        