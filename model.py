from torch import nn

class MNIST3dModel(nn.Module):
    
    def __init__(self, input_c=3, num_filters=8, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=input_c,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv3d(in_channels=num_filters,
                               out_channels=num_filters * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(in_channels=num_filters * 2,
                               out_channels=num_filters * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv3d(in_channels=num_filters * 4,
                               out_channels=num_filters * 8,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(4096, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)

        return x