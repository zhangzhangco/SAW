import gradio as gr
import torch
import json
import numpy as np
from model.model import Model
import torch.nn.functional as F
import tempfile
import os
import cv2
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import traceback  # 添加这一行
from model.backbone.pytorch_i3d import I3D
from video_features import I3DFeatureExtractor

# 改进配置文件加载
try:
    with open('json/config_Charades-STA_I3D_regression.json', 'r') as f:
        config = json.load(f)
        print("配置文件内容:", json.dumps(config, indent=2))
        
    # 确保配置格式正确
    if not isinstance(config, dict):
        raise ValueError("配置文件格式错误")
        
    # 统一配置结构
    if 'model_config' not in config:
        config = {'model_config': config}
    
    print("模型配置:", json.dumps(config['model_config'], indent=2))
except Exception as e:
    print(f"配置文件加载错误: {e}")
    raise

# 初始化模型
try:
    model = Model(config['model_config'])  # 确保使用 model_config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"模型将使用设备: {device}")
except Exception as e:
    print(f"模型初始化错误: {e}")
    raise

model.eval()

def translate_to_english(text):
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + text
    response = requests.get(url)
    if response.status_code == 200:
        translated_text = response.json()[0][0][0]
        print(f"翻译成功: '{text}' -> '{translated_text}'")
        return translated_text
    else:
        print(f"翻译请求失败，状态码：{response.status_code}")
        return text

def add_text_to_image(image, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    # 使用 Pillow 库来支持中文
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    font_path = "simsun.ttc"  # 请替换为您系统中的中文字体路径
    font = ImageFont.truetype(font_path, int(font_scale * 30))
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)

def load_text_encoder():
    """
    加载预训练的文本编码器
    """
    try:
        # 使用 BERT 作为文本编码器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_model = BertModel.from_pretrained('bert-base-uncased')
        text_model = text_model.to(device)
        text_model.eval()
        
        class TextEncoder(nn.Module):
            def __init__(self, bert_model, tokenizer):
                super(TextEncoder, self).__init__()
                self.bert = bert_model
                self.tokenizer = tokenizer
            
            def forward(self, text):
                # 对文本进行分词和编码
                inputs = self.tokenizer(text, 
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=20)  # 限制最大长度为20
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert(**inputs)
                    # 使用最后一层的隐藏状态
                    features = outputs.last_hidden_state  # [batch_size, seq_len, 768]
                    
                return features
                
        return TextEncoder(text_model, tokenizer)
        
    except Exception as e:
        print(f"加载文本编码器时出错: {e}")
        raise

def extract_i3d_features(video_path):
    """使用I3DFeatureExtractor提取特征"""
    try:
        extractor = I3DFeatureExtractor()
        features = extractor.extract_features(video_path)
        
        # 保存特征
        feature_path = video_path.replace('.mp4', '_i3d.npy')
        np.save(feature_path, features.cpu().numpy())
        return features
        
    except Exception as e:
        print(f"特征提取错误: {e}")
        raise

def load_i3d_features(video_path):
    """保持原有功能不变，增加特征不存在时的处理"""
    # 保持原有特征文件路径
    feature_path = "/mnt/d/charades/features/i3d_finetuned/" + os.path.basename(video_path).replace('.mp4', '.npy')
    
    try:
        # 首先尝试加载特征文件
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            return torch.from_numpy(features).to(device)  # 直接移到正确的设备
        else:
            print(f"特征文件不存在，开始提取特征: {feature_path}")
            features = extract_i3d_features(video_path)
            return features.to(device)  # 确保特征在��确的设备上
            
    except Exception as e:
        print(f"特征加载错误: {e}")
        print(f"特征路径: {feature_path}")
        print(f"特征形状: {features.shape if 'features' in locals() else 'unknown'}")
        raise

def load_projection_matrix():
    """初始化文本特征投影矩阵（768->300维）"""
    try:
        # 从配置文件获取目标维度
        target_dim = config['model_config']['embedding_dim']  # 300维
        
        # 创建并初始化投影矩阵
        projection = nn.Linear(768, target_dim)
        nn.init.orthogonal_(projection.weight)  # 使用正交初始化
        nn.init.zeros_(projection.bias)  # 将偏置初始化为0
        
        print(f"已初始化投影矩阵:")
        print(f"- 输入维度: 768 (BERT)")
        print(f"- 输出维度: {target_dim} (模型)")
        
        return projection.to(device)
        
    except Exception as e:
        print(f"投影矩阵初始化错误: {e}")
        traceback.print_exc()
        raise

def process_input(video_path, text_query):
    """处理输入的视频和文本"""
    try:
        # 1. 特征提取
        video_features = load_i3d_features(video_path)
        print(f"原始特征维度: {video_features.shape}")  # [99, 1, 1, 1024]
        
        # 调整视频特征维度为 [batch, time, feature_dim]
        if video_features.dim() == 4:  # [T, 1, 1, 1024]
            video_features = video_features.squeeze(1).squeeze(1)  # [T, 1024]
            video_features = video_features.unsqueeze(0)  # [1, T, 1024]
            
        # 重采样到指定的时间步长
        target_len = config['model_config']['segment_num']  # 75
        current_len = video_features.shape[1]
        if current_len != target_len:
            video_features = F.interpolate(
                video_features.transpose(1, 2),  # [1, 1024, T]
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [1, 75, 1024]
            
        video_features = video_features.to(device)
        print(f"处理后特征维度: {video_features.shape}")  # 应该是 [1, 75, 1024]
        print(f"视频特征维度检查:")
        print(f"- 形状: {video_features.shape}")
        print(f"- 类型: {video_features.dtype}")
        print(f"- 设备: {video_features.device}")

        # 2. 文本处理
        if any('\u4e00' <= char <= '\u9fff' for char in text_query):
            translated_query = translate_to_english(text_query)
            original_query = text_query
            print(f"将文本从'{original_query}'翻译为'{translated_query}'")
        else:
            translated_query = text_query
            original_query = text_query
            
        # 3. 文本编码
        text_encoder = load_text_encoder()
        projection = load_projection_matrix()
        
        with torch.no_grad():
            text_features = text_encoder(translated_query)
            print(f"BERT特征维度: {text_features.shape}")  # 应该是 [1, seq_len, 768]
            
            text_features = projection(text_features)
            print(f"投影后特征维度: {text_features.shape}")  # 应该是 [1, seq_len, 300]
            
            embedding_length = torch.tensor([text_features.size(1)], device=device)
            print(f"文本特征形状: {text_features.shape}")
            print(f"嵌入长度: {embedding_length}")
            
        return video_features, text_features, translated_query, original_query, embedding_length
        
    except Exception as e:
        print(f"输入处理错误: {e}")
        raise

def seconds_to_timestamp(seconds):
    """将秒数转换为 MM:SS 格式"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"

def extract_keyframes(video_path, start_time, end_time, query_text, confidence_score):
    """提取并处理关键帧"""
    frames_with_info = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return frames_with_info
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"视频FPS: {fps}")
        print(f"提取时间段: {start_time:.2f}s - {end_time:.2f}s")
        
        # 修改采样策略
        duration = end_time - start_time
        if duration <= 1.0:  # 短时间段
            num_frames = 3
            # 在中心点前后各取一帧
            center_time = (start_time + end_time) / 2
            frame_times = [
                max(start_time, center_time - 0.2),
                center_time,
                min(end_time, center_time + 0.2)
            ]
        else:  # 长时间段
            num_frames = min(max(3, int(duration * 2)), 7)  # 根据时长调整帧数
            # 确保包含开始和结束时间点
            frame_times = [start_time]
            if num_frames > 2:
                frame_times.extend(np.linspace(start_time, end_time, num_frames-2)[1:-1])
            frame_times.append(end_time)
            
        print(f"将提取 {len(frame_times)} 帧，时间点: {frame_times}")
        
        # 确保临时目录存在
        temp_dir = os.path.join(os.getcwd(), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"临时目录路径: {temp_dir}")
        
        for time in frame_times:
            frame_pos = int(time * fps)
            print(f"定位到帧位置: {frame_pos}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if ret:
                # 添加文本信息
                timestamp = seconds_to_timestamp(time)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 先转换颜色空间
                frame = process_frame(frame, timestamp, query_text)
                
                # 生成唯一的临时文件名
                temp_path = os.path.join(temp_dir, f"frame_{int(time*100):06d}.jpg")
                print(f"保存帧到: {temp_path}")
                
                # 确保图像是BGR格式用于保存
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 保存帧并验证
                success = cv2.imwrite(temp_path, save_frame)
                if success:
                    print(f"成功保存帧到: {temp_path}")
                    frames_with_info.append((temp_path, confidence_score))
                else:
                    print(f"保存帧失败: {temp_path}")
                    
            else:
                print(f"无法读取帧位置: {frame_pos}")
                
        print(f"成功提取了 {len(frames_with_info)} 帧")
        return frames_with_info
        
    except Exception as e:
        print(f"提取关键帧时出错: {e}")
        traceback.print_exc()
        return frames_with_info
        
    finally:
        if 'cap' in locals():
            cap.release()

def inference(video_path, text_query):
    """改进的推理函数"""
    progress = gr.Progress()
    progress(0, desc="开始处理...")
    
    try:
        # 输入验证
        if not os.path.exists(video_path):
            return [], {}, "视频文件不存在"  # 修改：返回三个值
        if not text_query.strip():
            return [], {}, "请输入查询文本"  # 修改：返回三个值
            
        # 获取实际视频时长
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        # 处理输入
        progress(0.2, desc="提取特征...")
        video_features, text_features, translated_query, original_query, embedding_length = process_input(video_path, text_query)
        
        print(f"\n推理前最终维度检查:")
        print(f"视频特征: {video_features.shape}")  # 应该是 [1, 1024, 99]
        print(f"文本特征: {text_features.shape}")   # 应该是 [1, seq_len, 300]
        print(f"嵌入长度: {embedding_length}")      # 应该是 tensor([seq_len])
        
        progress(0.4, desc="模型推理...")
        with torch.no_grad():
            # 获取预测和置信度
            predictions, confidence = model(
                video_features, 
                text_features, 
                embedding_length,
                gt_reg=None,
                score_nm=torch.ones(1, video_features.size(1)).to(device),
                mode='inference'
            )
            
            batch_idx = 0
            all_predictions = predictions[batch_idx]  # shape: [75, 2]
            all_confidence = confidence[batch_idx]
            
            # 对预测值进行归一化处理
            normalized_predictions = torch.clamp(all_predictions / all_predictions.max(), 0, 1)
            
            print("\n=== 预测值分析 ===")
            print(f"原始预测值范围: [{all_predictions.min().item():.4f}, {all_predictions.max().item():.4f}]")
            print(f"归一化后范围: [{normalized_predictions.min().item():.4f}, {normalized_predictions.max().item():.4f}]")
            
            # 收集所有有效预测
            valid_predictions = []
            for i in range(len(normalized_predictions)):
                # 将归一化的预测值转换为实际时间
                start = normalized_predictions[i][0].item() * video_duration
                end = normalized_predictions[i][1].item() * video_duration
                conf = all_confidence[i].item()
                
                # 确保时间在合理范围内
                if (0 <= start < video_duration and  # 开始时间在视频范围内
                    0 < end <= video_duration and    # 结束时间在视频范围内
                    end > start and                  # 结束时间大于开始时间
                    end - start >= 0.1 and          # 时长至少0.1秒
                    end - start <= video_duration * 0.5):  # 时长不超过视频长度的一半
                    
                    valid_predictions.append({
                        'index': i,
                        'start': start,
                        'end': end,
                        'confidence': conf
                    })

            print(f"视频总时长: {video_duration:.2f}秒")
            print(f"有效预测数量: {len(valid_predictions)}")
            if valid_predictions:
                print("前3个有效预测:")
                for pred in valid_predictions[:3]:
                    print(f"- 时间段: {pred['start']:.2f}s - {pred['end']:.2f}s (置信度: {pred['confidence']:.4f})")

            # 如果没有有效预测，返回错误
            if not valid_predictions:
                print("未找到有效的时间预测")
                return [], {}, "未找到有效的时间预测"
            
            # 按置信度排序并选择最佳预测
            valid_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            best_pred = valid_predictions[0]
            
            # 使用最佳预测
            start_time = best_pred['start']
            end_time = best_pred['end']
            conf_score = best_pred['confidence']

            print(f"\n预测结果分析:")
            print(f"有效预测数量: {len(valid_predictions)}")
            print(f"选择的预测: {best_pred}")
            print(f"时间范围: {start_time:.2f}s - {end_time:.2f}s")
            
            # 详细检查predictions的结构和内容
            print("\n=== 所有预测值分析 ===")
            batch_idx = 0
            all_predictions = predictions[batch_idx]  # shape: [75, 2]
            
            # 统计有效预测的数量（在视频时长范围内的）
            valid_preds = 0
            valid_pairs = []
            
            print(f"检查所有预测对（前10个）:")
            for i in range(min(10, len(all_predictions))):
                start, end = all_predictions[i][0].item(), all_predictions[i][1].item()
                conf = confidence[batch_idx][i].item()
                print(f"预测 {i}: start={start:.2f}s, end={end:.2f}s, conf={conf:.4f}")
                
                # 检查是否在合理范围内
                if 0 <= start <= video_duration and 0 <= end <= video_duration:
                    valid_preds += 1
                    valid_pairs.append((i, start, end, conf))
                    
            print(f"\n统计信息:")
            print(f"- 总预测数: {len(all_predictions)}")
            print(f"- 有效预测数: {valid_preds}")
            print(f"- 预测值范围: [{all_predictions.min().item():.2f}, {all_predictions.max().item():.2f}]")
            
            # 详细检查predictions的结构
            print("\n=== Predictions 诊断信息 ===")
            print(f"predictions shape: {predictions.shape}")
            print(f"predictions 类型: {predictions.dtype}")
            print(f"predictions 范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
            
            # 检查第一个batch的所有预测
            batch_idx = 0
            print("\n前几个预测结果:")
            for i in range(min(5, predictions.shape[1])):
                print(f"预测 {i}:")
                print(f"- 值: {predictions[batch_idx, i]}")
                print(f"- 置信度: {confidence[batch_idx, i].item():.4f}")
            
            # 获取最高置信度的预测
            batch_idx = 0
            max_conf_idx = torch.argmax(confidence[batch_idx])
            
            # 详细的预测信息打印
            print("\n=== 预测诊断信息 ===")
            print(f"预测张量形状: {predictions.shape}")
            print(f"置信度张量形状: {confidence.shape}")
            print(f"选择的预测索引: batch={batch_idx}, pred={max_conf_idx}")
            
            # 获取预测值
            pred_pair = predictions[batch_idx, max_conf_idx]  # 获取预测的时间对
            start_time = min(pred_pair[0].item(), pred_pair[1].item())  # 较小值作为开始时间
            end_time = max(pred_pair[0].item(), pred_pair[1].item())    # 较大值作为结束时间
            conf_score = confidence[batch_idx, max_conf_idx].item()
            
            print(f"\n原始预测值:")
            print(f"- start: {start_time:.4f}")
            print(f"- end: {end_time:.4f}")
            print(f"- confidence: {conf_score:.4f}")
            
            
            print(f"\n时间范围:")
            print(f"- 视频总时长: {video_duration:.2f}秒")
            print(f"- 开始时间: {start_time:.2f}秒")
            print(f"- 结束时间: {end_time:.2f}秒")
            print(f"- 时间段长度: {end_time - start_time:.2f}秒")
            
            # 确保时间在合理范围内
            start_time = max(0, min(start_time, video_duration))
            end_time = max(start_time, min(end_time, video_duration))
            
            # 转换为时间戳格式
            start_timestamp = seconds_to_timestamp(start_time)
            end_timestamp = seconds_to_timestamp(end_time)
            
            print(f"视频总时长: {video_duration:.2f}秒")
            print(f"预测时间段：{start_timestamp} 到 {end_timestamp}")
            
        # 置信度检查
        threshold = config.get('confidence_threshold', -1.0)
        if conf_score < threshold:
            return [], {}, f"预测置信度过低 ({conf_score:.2f} < {threshold})"  # 修改：返回三个值
            
        # 在提取关键帧之前添加检查
        print(f"\n准备提取关键帧:")
        print(f"视频路径: {video_path}")
        print(f"开始时间: {start_time:.2f}s")
        print(f"结束时间: {end_time:.2f}s")
        print(f"视频时长: {video_duration:.2f}s")
        
        if start_time >= end_time:
            return [], {}, f"无效的时间范围: {start_timestamp} - {end_timestamp}"
            
        if start_time >= video_duration or end_time <= 0:
            return [], {}, f"时间范围在视频范围之外: {start_timestamp} - {end_timestamp}"
        
        # 提取关键帧
        frame_paths = extract_keyframes(
            video_path, 
            start_time, 
            end_time, 
            original_query,
            conf_score
        )
        
        # 添加结果检查
        if not frame_paths:
            print("未能提取到任何关键帧！")
            return [], {}, "关键帧提取失败：未能提取到任何帧"
        
        # 整理输出结果
        frames_with_scores = [(path, f"置信度: {score:.2f}") for path, score in frame_paths]
        
        result = {
            "视频信息": get_video_info(video_path),
            "预测结果": {
                "开始时间": start_timestamp,
                "结束时间": end_timestamp,
                "置信度": f"{conf_score:.2f}",
                "查询文本": original_query
            }
        }
        
        progress(1.0, desc="完成")
        return (
            frames_with_scores,  # Gallery组件
            result,             # JSON组件
            f"定位成功！\n时间段：{start_timestamp} 到 {end_timestamp}"  # Textbox组件
        )
        
    except Exception as e:
        print(f"推理错误: {e}")
        traceback.print_exc()
        return [], {}, f"处理失败: {str(e)}"  # 确保错误情况也返回三个值

def get_video_info(video_path):
    """获取视频基本信息"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "状态": "无法打开视频文件"
            }
            
        # 获取视频基本属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        return {
            "分辨率": f"{width}x{height}",
            "帧率": f"{fps:.2f} fps",
            "总帧数": total_frames,
            "时长": f"{int(duration//60):02d}:{duration%60:05.2f}",
            "视频路径": video_path
        }
        
    except Exception as e:
        print(f"获取视频信息时出错: {e}")
        return {"状态": f"错误: {str(e)}"}
        
    finally:
        if 'cap' in locals():
            cap.release()

def process_frame(frame, timestamp, query_text):
    """处理视频帧，添加时间戳和查询文本"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 建议添加备选字体路径
        font_paths = [
            "simsun.ttc",
            "/usr/share/fonts/truetype/simsun.ttc",
            "/System/Library/Fonts/simsun.ttc"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 16)
                break
        else:
            font = ImageFont.load_default()
    except:
        # 如果失败，使用默认字体
        font = ImageFont.load_default()
    
    # 构建显示文本
    text = f"Time: {timestamp}\n{query_text}"
    
    # 计算文本区域
    text_bbox = draw.textbbox((0, 0), text, font=font)
    
    # 添加半透明背景
    draw.rectangle([(0, 0), (text_bbox[2] + 20, text_bbox[3] + 20)], 
                  fill=(0, 0, 0, 128))
    
    # 绘制文本
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    
    return np.array(img_pil)

# Gradio界面
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Video(label="输入视频"),
        gr.Textbox(label="文本查询", placeholder="在此输入动作描述...")
    ],
    outputs=[
        gr.Gallery(label="定位结果关键帧", show_label=True, columns=[5], height="auto"),
        gr.JSON(label="详细信息"),  # 使用 JSON 组件显示详细信息
        gr.Textbox(label="状态信息")
    ],
    title="时序句子定位演示 (Charades-STA)",
    description="上传视频并输入描述，系统将定位相关片段并显示关键帧。",
    examples=[
        ["sample_video1.mp4", "一个人在跳舞"],
        ["sample_video2.mp4", "有人在做饭"]
    ]
)

# 添加清理函数
def cleanup():
    temp_dir = os.path.join(os.getcwd(), "temp_frames")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except Exception as e:
                print(f"清理文件失败: {e}")

# 在程序退出时清理
if __name__ == "__main__":
    try:
        iface.launch()
    finally:
        cleanup()
