import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum

class GlobalTextPresentation(nn.Module):
    def __init__(self, text_dim):
        super(GlobalTextPresentation, self).__init__()
        # 创建一个线性层，用于转换文本特征
        self.W_txt = nn.Linear(text_dim, text_dim)

    def forward(self, fea_text, mask=None):
        # 调整文本特征的维度顺序：从 B*C2*L 变为 B*L*C2
        fea_text = fea_text.permute(0, 2, 1)  # B*L*C2
        # 应用线性变换得到注意力权重
        weight_text = self.W_txt(fea_text)  # B*L*C
        
        # 如果提供了掩码，则应用掩码（用于处理变长序列）
        if mask is not None:
            mask = mask.permute(0, 2, 1)
            weight_text = weight_text.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到归一化的注意力权重
        weight_text = weight_text.softmax(dim=1)
        
        # 将权重应用到文本特征上
        fea_text_global = fea_text * weight_text
        # 在序列维度上求和，得到全局文本表示
        fea_text_global = fea_text_global.sum(dim=1, keepdim=True).permute(0, 2, 1).unsqueeze(-1)  # B*C*1*1
        return fea_text_global

class GlobalAttention(nn.Module):
    def __init__(self, video_feature_dim, text_dim, global_attention_dim):
        super(GlobalAttention, self).__init__()
        self.scale = global_attention_dim ** -0.5

        # 创建查询(Q)、键(K)和值(V)的线性变换层
        self.Q = nn.Linear(video_feature_dim+text_dim+8, global_attention_dim)
        self.K = nn.Linear(text_dim, global_attention_dim)
        self.V = nn.Linear(text_dim, global_attention_dim)

    def forward(self, fea_video, fea_text):
        """
        :param fea_video: B*(C1+C2+8)*H*W
        :param fea_text: B*C2*1*1
        :param mask: B*1*L
        :return:
        """
        B, C1, H, W = fea_video.shape
        B, C2, _, _ = fea_text.shape
        # 调整视频和文本特征的形状
        fea_video = fea_video.view(B, C1, -1).permute(0, 2, 1)
        fea_text = fea_text.view(B, C2, -1).permute(0, 2, 1)

        # 计算查询、键和值
        q = self.Q(fea_video)
        k = self.K(fea_text)
        v = self.V(fea_text)

        # 计算注意力分数并应用缩放
        att = torch.matmul(q, k.permute(0, 2, 1)) * self.scale # B*HW*1
        att = att.softmax(-1)
        # 应用注意力权重得到输出
        out = torch.matmul(att, v)  # B*HW*C
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        return out


class LocalAttention(nn.Module):
    def __init__(self, video_feature_dim, text_dim, attention_dim):
        super(LocalAttention, self).__init__()
        self.scale = attention_dim ** -0.5

        self.Q = nn.Linear(video_feature_dim, attention_dim)
        self.K = nn.Linear(text_dim, attention_dim)
        self.V = nn.Linear(text_dim, attention_dim)

    def forward(self, fea_video, fea_text, mask):
        """
        :param fea_video: B*C*T*H*W
        :param fea_text: B*C*L
        :param mask: B*HW*L
        :return:
        """

        B, C, T, H, W = fea_video.shape
        fea_frames = fea_video.chunk(T, dim=2)
        fea_text = fea_text.permute(0, 2, 1)  # B*L*C
        outs = []
        for fea_frame in fea_frames:
            fea_frame = fea_frame.view(B, C, -1).permute(0, 2, 1)  # B*HW*C

            q = self.Q(fea_frame)
            k = self.K(fea_text)
            v = self.V(fea_text)

            att = torch.matmul(q, k.permute(0, 2, 1)) * self.scale  # B*HW*L
            if mask is not None:
                att = att.masked_fill(mask == 0, -1e9)
            att = att.softmax(-1)
            out = torch.matmul(att, v)  # B*HW*C
            out = out.permute(0, 2, 1).view(B, C, H, W).unsqueeze(2)
            outs.append(out)
        outs = torch.cat(outs, dim=2)
        return outs


class MuTan(nn.Module):
    def __init__(self, video_fea_dim, text_fea_dim, out_fea_dim, heads = 5):
        super(MuTan, self).__init__()

        self.heads = heads
        self.Wv = nn.ModuleList([nn.Conv2d(video_fea_dim+8, out_fea_dim, 1, 1) for i in range(heads)])
        self.Wt = nn.ModuleList([nn.Conv2d(text_fea_dim, out_fea_dim, 1, 1) for i in range(heads)])

    def forward(self, video_fea, text_fea, spatial):
        video_fea = torch.cat([video_fea, spatial], dim=1)
        fea_outs = []
        for i in range(self.heads):
            fea_v = self.Wv[i](video_fea)
            fea_v = torch.tanh(fea_v)  # B*C*H*W

            fea_t = self.Wt[i](text_fea)
            fea_t = torch.tanh(fea_t)  # B*C*1*1

            fea_out = fea_v * fea_t
            fea_outs.append(fea_out.unsqueeze(-1))
        fea_outs = torch.cat(fea_outs, dim=-1)
        fea_outs = torch.sum(fea_outs, dim=-1)
        mutan_fea = torch.tanh(fea_outs)
        mutan_fea = F.normalize(mutan_fea, dim=1)
        return mutan_fea

class RelevanceFilter(nn.Module):
    def __init__(self, text_fea_dim, video_fea_dim, attention_dim, groups=8, kernelsize=(1, 1, 1)):
        super(RelevanceFilter, self).__init__()
        assert text_fea_dim % groups == 0
        assert attention_dim % groups == 0
        self.groups = groups
        # 视频特征的3D卷积层
        self.Wv = nn.Conv3d(video_fea_dim, 2 * attention_dim, 1, 1)
        # 文本特征的线性层
        self.Wt = nn.Linear(text_fea_dim, attention_dim *
                            kernelsize[0] * kernelsize[1] * kernelsize[2])
        self.kernel_size = kernelsize

    def forward(self, video_fea, text_fea):
        # 应用视频特征卷积
        fea = self.Wv(video_fea)  # B*C*T*H*W
        B, C, T, H, W = video_fea.shape
        # 将特征分为key和value
        k, v = fea.chunk(2, dim=1)
        
        # 生成文本特征kernel
        kernel = self.Wt(text_fea)  # B*L*(C*K*K)
        kernel = repeat(kernel, 'b l (g c t h w) -> (b g l) c t h w',
                        t=self.kernel_size[0], h=self.kernel_size[1], w=self.kernel_size[2], g=self.groups)
        
        # 准备key特征
        k = repeat(k, 'b c t h w -> n (b c) t h w', n=1)
        
        # 应用3D卷积计算注意力
        att = F.conv3d(k, kernel, padding=(
            self.kernel_size[0]//2, self.kernel_size[1]//2, self.kernel_size[2]//2), groups=B*self.groups)
        att = rearrange(
            att, 'n (b g c) t h w -> (n b) g c t h w', b=B, g=self.groups)
        
        # 计算激活图
        active_map = att.mean(dim=1)
        
        # 重排value特征
        v = rearrange(v, 'b (g c) t h w -> b g c t h w', g=self.groups)
        
        # 应用注意力
        out = v * torch.sigmoid(att)
        out = rearrange(out, 'b g c t h w -> b (g c) t h w')
        
        # 准备输出的激活图
        maps = active_map.permute(2, 0, 1, 3, 4)
        maps = [maps[i] for i in range(T)]
        
        return maps, out