import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRLayer(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride, padding, is_2d=True):
        super().__init__()
        if is_2d:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        
        if is_2d:
            self.batch = nn.BatchNorm2d(out_channel)
        else:
            self.batch = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return x
    
class LowLevelFeature(nn.Module):
    def __init__(self, in_channels, C1, kernel_size, S1, S2, padding):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=C1, kernel_size=kernel_size, stride=S1, padding=padding)
        self.cbr = CBRLayer(in_channels=C1, out_channel=64, kernel_size=5, stride=S2, padding=2, is_2d=False)
        self.maxpool = nn.MaxPool1d(kernel_size=160 // (S1 * S2), stride=20)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbr(x)
        x = self.maxpool(x)
        return x
    
class HighLevelFeature(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.cbr1 = CBRLayer(in_channels=32, out_channel=64, kernel_size=(3, 3), stride=1, padding=1)
        self.cbr2 = CBRLayer(in_channels=64, out_channel=64, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.cbr3 = CBRLayer(in_channels=64, out_channel=128, kernel_size=(3, 3), stride=1, padding=1)
        self.cbr4 = CBRLayer(in_channels=128, out_channel=128, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.cbr5 = CBRLayer(in_channels=128, out_channel=256, kernel_size=(3, 3), stride=1, padding=1)
        self.cbr6 = CBRLayer(in_channels=256, out_channel=256, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.cbr7 = CBRLayer(in_channels=256, out_channel=512, kernel_size=(3, 3), stride=1, padding=1)
        self.cbr8 = CBRLayer(in_channels=512, out_channel=512, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.cbr9 = CBRLayer(in_channels=512, out_channel=num_classes, kernel_size=(1, 1), stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 4), stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.maxpool2(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        x = self.maxpool3(x)
        x = self.cbr5(x)
        x = self.cbr6(x)
        x = self.maxpool4(x)
        x = self.cbr7(x)
        x = self.cbr8(x)
        x = self.maxpool5(x)
        
        x = F.dropout(x, 0.2)

        x = self.cbr9(x)
        x = self.avgpool(x)

        return x

class AclNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super().__init__()

        C1 = 16
        S1 = 2
        S2 = 4

        self.LLF = LowLevelFeature(in_channels=input_channel, C1=C1, kernel_size=9, S1=S1, S2=S2, padding=4)
        self.HLF = HighLevelFeature(in_channels=1, num_classes=num_classes)

    def forward(self, x):
        x = self.LLF(x)
        x = torch.unsqueeze(x, 1)
        x = self.HLF(x)

        # x = F.softmax(x, 1)
        x = torch.squeeze(x, [2, 3])

        return x

