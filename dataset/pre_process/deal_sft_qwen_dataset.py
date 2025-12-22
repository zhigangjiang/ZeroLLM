import torch
from tqdm import tqdm
import json

# 指令文本处理
# 参考：https://github.com/QwenLM/Qwen/blob/main/finetune.py
def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    # prompt 模板
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # 不同的 tokenizer 需要特别定义
    # BOS
    im_start = tokenizer("<|im_start|>").input_ids
    # EOS
    im_end = tokenizer("<|im_end|>").input_ids
    # PAD
    IGNORE_TOKEN_ID = tokenizer.pad_token_id
    # 换行符
    nl_tokens = tokenizer('\n').input_ids
    # 角色标识符
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('human').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # 拼接多轮对话
    input_ids, targets = [], []
    for i in tqdm(range(len(sources))):
        source = sources[i]
        # 从 user 开始
        if source[0]["from"] != "human":
            source = source[1:]
        # 分别是输入和输出
        input_id, target = [], []
        # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        # system 不需要拟合
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        assert len(input_id) == len(target)
        # 依次拼接
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # user：<|im_start|>human\ninstruction【EOS】\n
            # assistant：<|im_start|>assistant\nresponse【EOS】\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>human':
                # user 不需要拟合
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant 需要拟合
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                print(role)
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # 最后进行 PAD
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    # print(input_ids)
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_path = "/root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

 

    with open("/root/projects/happy-llm/ZeroLLM/autodl-tmp/dataset/BelleGroup/train_3.5M_CN.json") as f:
        lst = [json.loads(line) for line in f.readlines()]

    # lst[0]

    # 测试一下
    res = preprocess([lst[0]["conversations"],lst[1]["conversations"]], tokenizer, 1024)
    print(res["input_ids"].shape)
    print(res["labels"].shape)
    print(res["input_ids"][0])
    print(res["labels"][0])
    print(tokenizer.decode(res["input_ids"][0]))
    print(tokenizer.decode(res["labels"][0]))
    print(tokenizer.decode(res["input_ids"][1]))
    print(tokenizer.decode(res["labels"][1]))