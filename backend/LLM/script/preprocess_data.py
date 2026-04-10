import json
import re
from bs4 import BeautifulSoup
import os

# 数据文件路径
data_dir = '/root/autodl-tmp/LLM/llm_dataset'
files = ['train.json', 'test.json']

# 收集所有数据和标签
all_data = []
all_labels = set()

# 读取数据文件
for file_name in files:
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        all_data.append(item)
                        # 收集标签
                        if item.get('summary'):
                            labels = [label.strip() for label in item['summary'].split(',')]
                            for label in labels:
                                if label:
                                    all_labels.add(label)
                    except json.JSONDecodeError:
                        print(f"Error decoding line: {line}")

print(f"Total data items: {len(all_data)}")
print(f"Total unique labels: {len(all_labels)}")
print(f"Labels: {sorted(all_labels)}")

# 文本清洗函数
def clean_text(text):
    # 去除HTML标签
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    # 标准化空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 处理特殊字符
    text = text.strip()
    return text

# 标签处理函数
def process_labels(summary):
    if not summary:
        return ['未知']
    labels = [label.strip() for label in summary.split(',')]
    # 过滤空标签
    labels = [label for label in labels if label]
    if not labels:
        return ['未知']
    return labels

# 预处理数据
processed_data = []
all_labels_sorted = sorted(all_labels)
labels_str = "、".join(all_labels_sorted)
for item in all_data:
    content = clean_text(item.get('content', ''))
    labels = process_labels(item.get('summary', ''))
    # 按照Qwen2.5要求的chat template格式组织数据
    user_content = "请根据以下金融文本内容,判断其所属的标签类别。标签包括:{labels}等。输出格式:标签1,标签2,...\n\n文本内容:{content}\n\n标签:".format(labels=labels_str, content=content)
    assistant_content = ",".join(labels)
    processed_item = {
        "messages": [
            {"role": "system", "content": "You are Qwen, a helpful assistant. You need to classify financial text into appropriate categories."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    processed_data.append(processed_item)

print(f"Processed data items: {len(processed_data)}")

# 划分数据集：训练集 + 测试集（9:1）
train_test_split = int(len(processed_data) * 0.9)
train_data = processed_data[:train_test_split]
test_data = processed_data[train_test_split:]

# 从训练集中分出500条作为验证集
dev_size = 500
train_data_final = train_data[:-dev_size]
dev_data = train_data[-dev_size:]

# 保存到文件
def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

output_dir = '/root/autodl-tmp/LLM/llm_dataset_processed'
os.makedirs(output_dir, exist_ok=True)

save_data(train_data_final, os.path.join(output_dir, 'train.json'))
save_data(dev_data, os.path.join(output_dir, 'dev.json'))
save_data(test_data, os.path.join(output_dir, 'test.json'))

print(f"Saved processed data to {output_dir}")
print(f"Train size: {len(train_data_final)}")
print(f"Dev size: {len(dev_data)}")
print(f"Test size: {len(test_data)}")
