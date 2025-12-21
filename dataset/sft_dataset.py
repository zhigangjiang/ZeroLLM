from torch.utils.data import Dataset
import torch
import json
import numpy as np

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # self.data = f.readlines()
            for line in f:
                self.data.append(line)
                if len(self.data) >= 1000:  # 只加载前100万行，防止内存占用过高
                    break
        self.a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']
        self.im_end_id = self.tokenizer("<|im_end|>")['input_ids'][0]

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        # a_sequence = [3, 1074, 537, 500, 203]  # <|im_start|>assistant\n
        a_sequence = self.a_sequence
        im_end_id = self.im_end_id  # <|im_end|> 的id
        
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个4, 4 为 <|im_end|> EOS id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == im_end_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer_path = "src/tokenizer_k"  # 替换为保存tokenizer的路径
    data_path = "./src/dataset/BelleGroup/BelleGroup_sft.jsonl"  # 替换为你的数据文件路径
    # 示例用法
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = SFTDataset(data_path=data_path, tokenizer=tokenizer, max_length=512)

    # 获取一个样本
    X, Y, loss_mask = dataset[0]
    print("Input IDs:", X)
    print("Labels:", Y)
    print("Loss Mask:", loss_mask)