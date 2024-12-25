import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, cls_num, feature_num):
        super(CenterLoss,self).__init__()
        self.cls_num = cls_num
		# 随机10个center
        self.center = nn.Parameter(torch.randn(cls_num, feature_num), requires_grad=True)
    
    def forward(self, feature, _target):
        
        feature = F.normalize(feature)				# 对特征做归一化
		# 将center广播成特征点那么多个，每一个特征对应一个center
        centre = self.center.cuda().index_select(dim=0, index=_target.long())

		# 统计每个类别有多少的数据
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num-1)
		# 将每个类别的统计数量广播，每个数据对应一个该类的总数，好做计算
        count = counter[_target.long()]
        centre_dis = feature - centre				# 做差，每个特征到它中心点的距离
        pow_ = torch.pow(centre_dis, 2)				# 平方
        sum_1 = torch.sum(pow_, dim=1)				# 横向求和，每个类别的距离总和
        dis_ = torch.div(sum_1, count.float())		# 类别差，每个类别的差除以该类的总量，得到该类均差
		# sqrt_ = torch.sqrt_(dis_)					# 开方
        sum_2 = torch.sum(dis_)						# 求总差，所有类别的差
        res = sum_2 / 2.0							# 乘：lambda / 2，
        
        return res

if __name__ == '__main__':
    center_loss_cuda = CenterLoss(cls_num=2, feature_num = 2)
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.long)
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.long)
    loss = center_loss_cuda(data,label)
    print(loss)
