from torch.utils.data import Dataset
import torch
import json
import numpy as np
from tqdm import tqdm

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # self.data = f.readlines()
            for line in tqdm(f):
                self.data.append(line)
                if len(self.data) >= 10000000:  # 只加载前1000万行，防止内存占用过高
                    break


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer_path = "src/tokenizer_k"  # 替换为保存tokenizer的路径
    data_path = "./src/dataset/seq_monkey_datawhale.jsonl"  # 替换为你的数据文件路径
    # 示例用法
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = PretrainDataset(data_path=data_path, tokenizer=tokenizer, max_length=512)

    # 获取一个样本
    X, Y, loss_mask = dataset[0]
    print("Input IDs:", X)
    print("Labels:", Y)
    print("Loss Mask:", loss_mask)