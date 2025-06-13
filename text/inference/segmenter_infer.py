import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import BertTokenizerFast
from model.bert_segmenter import BertSegmenter
import numpy as np

class Segmenter:
    def __init__(self, model_path, model_name='/root/.cache/modelscope/hub/models/google-bert/bert-base-chinese', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertSegmenter(model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def segment(self, text, threshold=0.5):
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, return_offsets_mapping=True)
        print(f"Encoding result: {encoding}")
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()  # 取切割概率
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        cut_points = np.where(probs > threshold)[0]
        # 根据cut_points切割文本
        offsets = encoding['offset_mapping'][0].cpu().numpy() if 'offset_mapping' in encoding else None
        segments = []
        last = 0
        # 由于tokens可能包含特殊字符如[CLS], [SEP]，使用offset_mapping进行字符级别切割
        for idx in cut_points:
            if idx >= len(offsets): # 避免索引越界
                continue
            current_token_end_char_idx = offsets[idx][1]
            if current_token_end_char_idx > last:
                segments.append(text[last:current_token_end_char_idx])
                last = current_token_end_char_idx
        
        # 添加最后一个片段
        if last < len(text):
            segments.append(text[last:])

        return [seg.strip() for seg in segments if seg.strip()]

if __name__ == '__main__':
    model_path = '/root/text/model/bert_segmenter_epoch5.pt'
    segmenter = Segmenter(model_path)
    text = '番茄在南美热带地区原是多年生植物，但在温带则为一年生作物。番茄的植株由根、茎、叶、花、果实及种子所组成，其特征特性分述如下。'
    segments = segmenter.segment(text)
    for i, seg in enumerate(segments):
        print(f'[{i}] {seg}') 