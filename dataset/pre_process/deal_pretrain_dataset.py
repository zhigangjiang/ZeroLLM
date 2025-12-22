import os
import json
from tqdm import tqdm

# pretrain_data 为运行download_dataset.sh时，下载的pretrain_data本地路径
pretrain_data = '/root/autodl-tmp/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = '/root/autodl-tmp/seq_monkey_datawhale.jsonl'


# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):  # 添加行级别的进度条
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
