# Model part
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits = self._tower(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
    
class MahjongModel(nn.Module):
    def __init__(self, feature_dim=141, action_dim=235):
        super(MahjongModel, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(feature_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.att_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.att_bn = nn.BatchNorm2d(1)
        self.att_relu = nn.ReLU(inplace=True)
        self.att_softmax = nn.Softmax(dim=-1)
        
        self.fc1 = nn.Linear(128*4*9, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_relu1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(256, 256)
        self.fc_relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        x = input_dict["obs"]["observation"].float()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        feature = self.relu3(self.bn3(self.conv3(x)))
        
        att = self.att_softmax(self.att_relu(self.att_bn(self.att_conv(feature))))
        feature = feature * att
        
        x = feature.view(feature.size(0), -1)
        x = self.fc_relu1(self.fc_bn1(self.fc1(x)))
        x = self.fc_relu2(self.fc2(x))
        x = self.fc3(x)
        
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask
    

class ResMahjongModel(nn.Module):
    def __init__(self, feature_dim=141, action_dim=235):
        super(ResMahjongModel, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        self.conv1 = nn.Conv2d(feature_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.att_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.att_bn = nn.BatchNorm2d(1)
        self.att_relu = nn.ReLU(inplace=True)
        self.att_softmax = nn.Softmax(dim=-1)
        
        self.fc1 = nn.Linear(128*4*9, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_relu1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(256, 256)
        self.fc_relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        x = input_dict["obs"]["observation"].float()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        feature = self.relu2(self.bn2(self.conv2(x)))
        
        att = self.att_softmax(self.att_relu(self.att_bn(self.att_conv(feature))))
        feature = feature * att
        
        x = feature.view(feature.size(0), -1)
        x = self.fc_relu1(self.fc_bn1(self.fc1(x)))
        x = self.fc_relu2(self.fc2(x))
        x = self.fc3(x)
        
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask