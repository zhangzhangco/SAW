import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, norm_type='GroupNorm'):
        """
        ASPP模块的单个分支
        :param inplanes: 输入通道数
        :param planes: 输出通道数
        :param rate: 空洞卷积的扩张率
        :param norm_type: 归一化类型，默认为GroupNorm
        """
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        # 选择归一化类型
        if norm_type=='GroupNorm':
            self.bn = nn.GroupNorm(8, planes)  # 使用8个组的GroupNorm
        else:
            self.bn = nn.BatchNorm2d(planes)
        # 空洞卷积层
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, 
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates, norm_type='GroupNorm'):
        """
        ASPP模块的主体结构
        :param inplanes: 输入通道数
        :param planes: 输出通道数
        :param rates: 不同分支的空洞卷积扩张率列表
        :param norm_type: 归一化类型，默认为GroupNorm
        """
        super(ASPP, self).__init__()

        # 创建4个不同扩张率的ASPP分支
        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0], norm_type=norm_type)
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1], norm_type=norm_type)
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2], norm_type=norm_type)
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3], norm_type=norm_type)

        self.relu = nn.ReLU()

        # 全局平均池化分支
        if norm_type=='GroupNorm':
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                nn.GroupNorm(8, planes),
                nn.ReLU()
            )
            self.bn1 = nn.GroupNorm(8, planes)
        else:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )
            self.bn1 = nn.BatchNorm2d(planes)

        # 最终的1x1卷积，用于融合所有分支的特征
        self.conv1 = nn.Conv2d(planes*5, planes, 1, bias=False)
        self.__init_weight()

    def __init_weight(self):
        """初始化模块的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 通过4个ASPP分支
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        # 全局平均池化分支
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 拼接所有分支的输出
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # 最终的1x1卷积处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
