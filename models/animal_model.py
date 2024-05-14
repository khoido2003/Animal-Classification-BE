
import torch.nn as nn

class AnimalModel(nn.Module):
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self._make_block(in_channels=3, out_channels=16)
        self.conv2 = self._make_block(in_channels=16, out_channels=32)
        self.conv3 = self._make_block(in_channels=32, out_channels=64)
        self.conv4 = self._make_block(in_channels=64, out_channels=64)
        self.conv5 = self._make_block(in_channels=64, out_channels=128)
        self.linear_1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(7*7*128, 1024),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.linear_3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x