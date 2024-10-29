import torch
import torch.nn as nn
# 修改导入语句，使用绝对导入
from model.backbone.pytorch_i3d import I3D
import cv2
import numpy as np

class I3DFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.i3d = I3D(num_classes=400)  # 移除 in_channels 参数
        # 加载预训练模型
        self.i3d.load_state_dict(torch.load('model/pretrained/rgb_imagenet.pt'))
        self.i3d = self.i3d.to(self.device)
        self.i3d.eval()
        
    def extract_frames(self, video_path, target_fps=25):
        """均匀采样视频帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        sample_interval = video_fps / target_fps
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval < 1:
                # 预处理帧
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        return np.array(frames)
        
    def extract_features(self, video_path):
        """提取 I3D 特征"""
        try:
            # 提取并预处理帧
            frames = self.extract_frames(video_path)
            if len(frames) == 0:
                raise ValueError("未能从视频中提取到帧")
                
            # 转换为张量
            frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
            frames = frames.unsqueeze(0)  # [1, num_frames, 3, 224, 224]
            
            with torch.no_grad():
                frames = frames.to(self.device)
                features = self.i3d.extract_features(frames)
                # features 应该是 [1, 1024, T, 1, 1]
                features = features.squeeze(-1).squeeze(-1)  # 移除空间维度
                # 现在是 [1, 1024, T]
                features = features.permute(0, 2, 1)  # 变成 [1, T, 1024]
                
            print(f"I3D特征维度检查: {features.shape}")  # 用于调试
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            raise
