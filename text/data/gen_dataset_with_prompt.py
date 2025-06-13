import os
import json
from zhipuai import ZhipuAI
import re
import time

API_KEY = '8f99d8f2317a455fbf5e2133f1586c64.aekqGpaMyoCMmmPQ'

client = ZhipuAI(api_key=API_KEY)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "你是一个文本语义分析专家。请根据语义完整性将以下文本切割成若干段落，每个段落应表达一个完整的语义单元。输出格式为JSON，包含'text'和'segments'字段。"
}


def generate_segmented_data(input_texts, output_path):
    results = []
    for text in input_texts:
        messages = [SYSTEM_PROMPT, {"role": "user", "content": text}]
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
            top_p=0.7,
            temperature=0.95,
            max_tokens=1024,
            stream=False
        )
        try:
            content = response.choices[0].message.content
            print("API返回内容：", content)
            # 去除代码块标记
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            data = json.loads(content)
            results.append(data)
            # 每次成功解析后立即写入文件，防止全部失败时无输出
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing text: {text}\n{e}")
    # 如果全部失败，写入空列表
    if not results:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data/raw/sample_texts.txt')
    output_file = os.path.join(base_dir, 'data/processed/segmented_data.json')
    with open(input_file, 'r', encoding='utf-8') as f:
        # 放宽正文过滤条件
        def is_content(line):
            line = line.strip()
            if len(line) < 20:
                return False
            if re.search(r'[。！？]', line):
                return True
            return False
        input_texts = [line.strip() for i, line in enumerate(f) if is_content(line)][:200]
    results = []
    for text in input_texts:
        success = False
        for attempt in range(3):
            try:
                messages = [SYSTEM_PROMPT, {"role": "user", "content": text}]
                response = client.chat.completions.create(
                    model="glm-4-plus",
                    messages=messages,
                    top_p=0.7,
                    temperature=0.95,
                    max_tokens=1024,
                    stream=False
                )
                content = response.choices[0].message.content
                print("API返回内容：", content)
                # 去除代码块标记
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                data = json.loads(content)
                results.append(data)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                success = True
                break
            except Exception as e:
                print(f"Error processing text (attempt {attempt+1}): {text}\n{e}")
                time.sleep(1)
        time.sleep(1)  # 每次API调用后sleep 1秒，防止限流
    if not results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main() 