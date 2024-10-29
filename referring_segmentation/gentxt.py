import json
import csv

def generate_annotation_from_json(json_file_path, output_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # 创建并写入 CSV 文件
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video_name', 'description'])  # 写入标题行
        
        for video_name, video_data in data.items():
            # 假设每个视频只有一个实例描述
            description = next(iter(video_data['instances'].values()))
            writer.writerow([video_name, description])

    print(f"注释文件已生成: {output_file_path}")

if __name__ == '__main__':
    json_file_path = './data/jhmdb_sentences_test.json'
    output_file_path = 'jhmdb_annotation.txt'
    
    generate_annotation_from_json(json_file_path, output_file_path)

