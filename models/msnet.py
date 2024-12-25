import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 输入x = (batch, 1, 1024)的模型，输出结果为(64, 4)
class MSNet(nn.Module):
    def __init__(self,in_features = 256*4,out_features=256):
        super(MSNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= 64, kernel_size= 8, stride= 2, padding= 4)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.BN = nn.BatchNorm1d(num_features=1)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv3_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.pool3_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool5_1 = nn.MaxPool1d(kernel_size=2 , stride=2)
        self.conv5_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool5_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv5_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        # self.pool5_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv7_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.pool7_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv7_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool7_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv7_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3)
        # self.pool7_3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.pool9_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv9_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=1, padding=4)
        self.pool9_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv9_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, stride=1, padding=4)
        # self.pool9_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1)
        self.out_layer_1  = nn.Linear(in_features, out_features)
        self.out_layer_2 = nn.Linear(out_features, 16)  ##这里的4096是计算出来的
        # self.softmax = nn.Softmax()
        
    def forward(self, x):
        # x = self.BN(x)
        x = self.conv1(x)  ## x:Batch, 1, 1024
        x = self.pool1(x)
        # kernel_size为3
        x1 = self.conv3_1(x)
        x1 = self.pool3_1(x1)
        x1 = self.conv3_2(x1)
        x1 = self.pool3_2(x1)
        # x1 = self.conv3_3(x1)
        # x1 = self.pool3_3(x1)
        
        # kernel_size为5
        x2 = self.conv5_1(x)
        x2 = self.pool5_1(x2)
        x2 = self.conv5_2(x2)
        x2 = self.pool5_2(x2)
        # x2 = self.conv5_3(x2)
        # x2 = self.pool5_3(x2)
        
        # kernel_size为7
        x3 = self.conv7_1(x)
        x3 = self.pool7_1(x3)
        x3  = self.conv7_2(x3)
        x3 = self.pool7_2(x3)
        # x3 = self.conv7_3(x3)
        # x3 = self.pool7_3(x3)

        # kernel_size为9
        # x4 = self.conv9_1(x)
        # x4 = self.pool9_1(x4)
        # x4 = self.conv9_2(x4)
        # x4 = self.pool9_2(x4)
        # x4 = self.conv9_3(x4)
        # x4 = self.pool9_3(x4)

        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        
        # 池化层
        x1 = self.pool2(x1)
        x2 = self.pool2(x2)
        x3 = self.pool2(x3)
        # x4 = self.pool2(x4)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        
        # flatten展平
        Batch, Channel, Length = x1.size()
        x1 = x1.view(Batch, -1)
        Batch, Channel, Length = x2.size()
        x2 = x2.view(Batch, -1)
        Batch, Channel, Length = x3.size()
        x3 = x3.view(Batch, -1)
        # Batch, Channel, Length = x4.size()
        # x4 = x4.view(Batch, -1)
        #将3个尺度提取到的特征连接在一起d
        # x0 = torch.cat((x1, x2, x3), dim=1) # torch.Size([8, 256]) torch.Size([8, 256]) torch.Size([8, 256]) torch.Size([8, 256])
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        # out = self.out_layer(x0) 
		# # # 全连接层
        # x1 = self.fc(x1)
        # return out
        return x1,x2,x3
