import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.attention import LocalAttention, RelevanceFilter


class TCN(nn.Module):
    def __init__(self, text_dim, inchannel, hidden_channel, outchannel, layers=8, padding_type='zero', with_local_attention=True, conv_type='3D', local_attention_type='relevance_filter', groups=8, norm_type='GroupNorm'):
        super(TCN, self).__init__()
        # 初始化TCN（时序卷积网络）的各种参数和模块
        # 这种结构设计用于有效处理视频和文本的多模态信息，同时捕获短期和长期的时序依赖关系
        self.padding_type = padding_type
        self.with_local_attention = with_local_attention
        self.local_attention_type = local_attention_type
        self.conv_time = nn.ModuleList()
        self.conv_spatial = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        self.dilations = []
        self.local_attention = nn.ModuleList()
        # self.global_txt_W = nn.ModuleList()
        for i in range(layers):
            # 使用指数增长的膨胀率，以指数级增加感受野
            dilation = torch.pow(torch.tensor(2), i)
            dilation = int(dilation)
            self.dilations.append(dilation)
            
            # 添加局部注意力模块，用于融合视频和文本特征
            if with_local_attention:
                if local_attention_type == 'attention':
                    self.local_attention.append(LocalAttention(inchannel, text_dim, inchannel))
                else:
                    self.local_attention.append(RelevanceFilter(text_dim, inchannel, inchannel, groups=groups))
            else:
                self.local_attention.append(nn.Identity())

            # 根据conv_type选择不同的卷积结构
            # 3D卷积同时处理空间和时间维度，而分离模式先处理空间再处理时间
            if conv_type == '3D':
                self.conv_spatial.append(nn.Identity())
                if norm_type == "GroupNorm":
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (3, 3, 3), 1, (0, 1, 1), (dilation, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True))
                        )
                else:
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (3, 3, 3), 1, (0, 1, 1), (dilation, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True))
                        )

            else:
                if norm_type == "GroupNorm":
                    self.conv_spatial.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (1, 3, 3), 1, (0, 1, 1), (1, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(hidden_channel, hidden_channel, (3, 1, 1), (1, 1, 1), (0, 0, 0), (dilation, 1, 1), bias=False),
                            nn.GroupNorm(8, hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    self.conv_spatial.append(
                        nn.Sequential(
                            nn.Conv3d(inchannel, hidden_channel, (1, 3, 3), 1, (0, 1, 1), (1, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
                    self.conv_time.append(
                        nn.Sequential(
                            nn.Conv3d(hidden_channel, hidden_channel, (3, 1, 1), (1, 1, 1), (0, 0, 0), (dilation, 1, 1), bias=False),
                            nn.BatchNorm3d(hidden_channel),
                            nn.ReLU(inplace=True)
                        )
                    )
            if norm_type == "GroupNorm":
                self.conv_convert.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_channel, outchannel, 1, 1, bias=False),
                        nn.GroupNorm(8, outchannel)
                    )
                )
            else:
                self.conv_convert.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_channel, outchannel, 1, 1, bias=False),
                        nn.BatchNorm3d(outchannel)
                    )
                )
        self.__init_weight()

    def _create_conv_block(self, in_channels, out_channels, kernel_size, dilation, norm_type, use_relu=True):
        # 创建卷积块，包括卷积、归一化和激活函数
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, 1, (0, 1, 1), (dilation, 1, 1), bias=False),
            nn.GroupNorm(8, out_channels) if norm_type == "GroupNorm" else nn.BatchNorm3d(out_channels)
        ]
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def __init_weight(self):
        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fea, fea_text, mask_local):
        fea_text = fea_text.permute(0, 2, 1)  # B*L*C
        maps_layers = []
        for i in range(len(self.conv_time)):
            res0 = fea

            # 应用局部注意力，融合视频和文本特征
            if self.with_local_attention:
                if self.local_attention_type == 'attention':
                    fea = self.local_attention[i](fea, fea_text, mask_local)
                else:
                    maps, fea = self.local_attention[i](fea, fea_text)
                    maps_layers.append(maps)
                fea = res0 + fea  # 残差连接

            res1 = fea
            fea = self.conv_spatial[i](fea)

            # 应用不同类型的填充，处理时序边界问题
            if self.padding_type == 'circle':
                fea = circle_padding(self.dilations[i], fea)
            elif self.padding_type == 'zero':
                fea = F.pad(fea, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='constant', value=0)
            else:
                fea = F.pad(fea, (0, 0, 0, 0, self.dilations[i], self.dilations[i]), mode='circular')

            fea = self.conv_time[i](fea)  # 时间卷积，捕获时序依赖
            fea = self.conv_convert[i](fea)  # 调整通道数
            fea = fea + res1  # 残差连接，有助于训练更深的网络
        return fea, maps_layers


def circle_padding(padding, feature):
    # 实现循环填充，处理时序边界问题
    length_times = feature.shape[2]
    index = list(range(0, length_times)) + list(range(length_times - 2, 0, -1))
    total_num = 2 * padding + length_times
    num_c = padding // len(index)
    if num_c * len(index) < padding:
        num_c = num_c + 1
    expand_number = num_c * len(index) - padding
    index_f = []
    for n in range(num_c):
        index = index + index + index
    for i in range(expand_number, expand_number + total_num):
        index_f.append(index[i])

    feas = []
    for idf in index_f:
        feas.append(feature[:, :, idf, :, :].unsqueeze(2))
    feas = torch.cat(feas, dim=2)
    return feas
