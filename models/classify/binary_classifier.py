import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class BinaryClassifier(nn.Module):
    def __init__(self, input_channel: int):
        super(BinaryClassifier, self).__init__()
        self.transform = nn.Sequential(
            ConvModule(
                in_channels=input_channel,
                out_channels=input_channel*2,
                kernel_size=3,
                padding='same',
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU')),
            nn.MaxPool2d(kernel_size=2,stride=2),
            ConvModule(
                in_channels=input_channel*2,
                out_channels=input_channel,
                kernel_size=3,
                padding='same',
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU')),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
                nn.Linear(input_channel, input_channel//2),
                nn.ReLU(),
                nn.Linear(input_channel//2, 1),
            )
    
    def forward(self, image_embed):
        '''
        Args:
            image_embed: Tensor, shape is (bs, 256, 64, 64)
        '''
        image_embed = self.transform(image_embed)
        image_embed = torch.mean(image_embed, dim=(-1,-2))
        logit = self.classifier(image_embed)
        return logit