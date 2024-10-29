import gradio as gr
import torch
import torchtext
import cv2
import numpy as np
import json
from model.model import Model
from torchvision import transforms
import torch.nn.functional as F
import tempfile
import os
from pathlib import Path
import requests

# 加载配置
with open('json/config_a2d_sentences.json', 'r') as f:
    config = json.load(f)

# 初始化模型
model_config = config['model_config']
model = Model(model_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    checkpoint = torch.load('./model/pretrained/best_deeplabv3plus_resnet101_voc_os16.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("模型加载成功。")
except Exception as e:
    print(f"加载模型时出错: {e}")
    raise

model = model.to(device)
model.eval()

# 全局加载GloVe词向量
try:
    from torchtext.vocab import GloVe
    vocab = GloVe(name='6B', dim=300)
    print("GloVe词向量加载成功。")
except Exception as e:
    print(f"加载GloVe词向量时出错: {e}")
    raise

def translate_to_english(text):
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + text
    response = requests.get(url)
    if response.status_code == 200:
        translated_text = response.json()[0][0][0]
        print(f"翻译成功: '{text}' -> '{translated_text}'")  # 添加这行来打印翻译结果
        return translated_text
    else:
        print(f"翻译请求失败，状态码：{response.status_code}")
        return text

def process_input(video_path, text_query):
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        # 读取视频帧
        frames = []
        while len(frames) < 300:  # 限制最大帧数
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError("视频中没有找到帧")

        # 均匀采样8帧
        step = max(len(frames) // 8, 1)
        sampled_frames = frames[::step][:8]
        
        if len(sampled_frames) < 8:
            sampled_frames = sampled_frames * (8 // len(sampled_frames) + 1)
        sampled_frames = sampled_frames[:8]

        # 预处理帧
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config['data_config']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        processed_frames = torch.stack([preprocess(frame) for frame in sampled_frames])

        # 检测输入是否为中文
        if any('\u4e00' <= char <= '\u9fff' for char in text_query):
            # 如果是中文，调用翻译服务
            text_query = translate_to_english(text_query)
            print(f"翻译后的查询: {text_query}")
        else:
            print("输入为英文，无需翻译")

        # 处理文本查询
        words = text_query.lower().split()
        if len(words) == 0:
            raise ValueError("文本查询为空")

        # 获取词嵌入
        word_embeddings = [vocab[word] for word in words]
        max_length = config['data_config']['max_embedding_length']
        if len(word_embeddings) > max_length:
            word_embeddings = word_embeddings[:max_length]
        else:
            padding = [torch.zeros(300) for _ in range(max_length - len(word_embeddings))]
            word_embeddings.extend(padding)
        
        word_embeddings = torch.stack(word_embeddings)

        return processed_frames.unsqueeze(0), word_embeddings.unsqueeze(0), len(words), frames
    except Exception as e:
        print(f"处理输入时出错: {e}")
        raise

def visualize_output(frames, output):
    try:
        # 将输出调整为原始帧的大小
        mask = F.interpolate(output, size=(frames[0].shape[0], frames[0].shape[1]), mode='bilinear', align_corners=False)
        mask = mask.squeeze().cpu().numpy()
        
        # 创建彩色遮罩
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask > 0.5] = [0, 255, 0]  # 绿色

        # 将遮罩叠加到原始帧上
        result_frames = []
        for frame in frames:
            overlay = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
            result_frames.append(overlay)

        return np.array(result_frames)
    except Exception as e:
        print(f"可视化输出时出错: {e}")
        raise

def inference(video_path, text_query):
    try:
        # 处理输入
        frames, text_embedding, embedding_length, original_frames = process_input(video_path, text_query)
        frames = frames.to(device)
        text_embedding = text_embedding.to(device)
        
        # 模型推理
        with torch.no_grad():
            output = model(frames, text_embedding, [embedding_length])
        
        predicted_mask = torch.sigmoid(output[0])
        
        # 可视化结果
        result_frames = visualize_output(original_frames, predicted_mask)
        
        # 保存结果为视频
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            out_path = temp_file.name
        
        original_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, original_fps, (result_frames.shape[2], result_frames.shape[1]))
        for frame in result_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        return out_path
    except Exception as e:
        print(f"推理过程中出错: {e}")
        return None
    finally:
        # 清理临时文件
        if 'out_path' in locals():
            try:
                os.remove(out_path)
            except Exception as e:
                print(f"删除临时文件时出错: {e}")

# Gradio接口
iface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Video(label="输入视频"),
        gr.Textbox(label="文本查询", placeholder="在此输入对象描述...")
    ],
    outputs=[
        gr.Video(label="分割结果"),
        gr.Textbox(label="处理时间")
    ],
    title="视频指代分割演示",
    description="上传一个视频并输入您想要分割的对象的文本描述。",
    examples=[
        ["sample_video.mp4", "一辆红色的车"],
        ["sample_video2.mp4", "一个穿蓝色衬衫的人"]
    ],
    theme="huggingface",
    css="body { background-color: #f0f0f0; }"
)

if __name__ == "__main__":
    iface.launch()
