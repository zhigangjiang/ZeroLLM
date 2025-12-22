from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain


# 预训练一般将文本拼接成固定长度的文本段
def group_texts_(examples):



    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
    print("group texts input examples length%d after_group size%d"%(len(examples['input_ids']),len(result["input_ids"])))
    result["labels"] = result["input_ids"].copy()
    return result


def group_texts(examples):
    # examples 是一个字典，包含了一批数据（batch）
    # 结构类似：{'input_ids': [[1,2,3], [4,5], ...], 'attention_mask': [...]}
    
    # 1. 拼接 (Concatenation)
    # chain(*examples[k]) 把列表里的所有子列表“拉直”连成一条长龙
    # 比如 input_ids 原本是 [[1,2], [3,4,5]]，变成 [1,2,3,4,5]
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
    # 2. 计算总长度
    # 获取拼接后的总长度，比如总共有 10050 个 token
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 3. 丢弃尾部 (Truncation)
    # 如果总长度不是 block_size 的整数倍，就把多余的尾巴切掉
    # 假设 block_size = 1000，total_length = 10050
    # (10050 // 1000) * 1000 = 10000
    # 这样能保证切分时每块都是完整的 1000 长度
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # 4. 切分 (Chunking)
    # 按照 block_size 进行切分
    # range(0, 10000, 1000) -> 0, 1000, 2000...
    # t[0:1000], t[1000:2000]...
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # 打印日志：看看拼接前有多少条，拼接后变成了多少条
    # 通常拼接后条数会变少（因为很多短句拼成了一个长句），但每条的信息量变大了
    print("group texts input examples length%d after_group size%d"%(len(examples['input_ids']),len(result["input_ids"])))
    
    # 5. 设置标签 (Labels)
    # 在预训练（Causal Language Modeling）中，输入和标签是一样的
    # 模型任务是：根据 input_ids[0:-1] 预测 input_ids[1:]
    # HuggingFace 的 Trainer 会自动处理移位，所以这里直接复制即可
    result["labels"] = result["input_ids"].copy()
    
    return result





def tokenize_function(examples):
    # 使用预先加载的 tokenizer 进行分词
    output = tokenizer([item for item in examples["text"]])
    return output


if __name__ == "__main__":
    model_path = "/root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B"
    data_files='/root/projects/happy-llm/ZeroLLM/autodl-tmp/seq_monkey_datawhale_small.jsonl'


    tokenizer = AutoTokenizer.from_pretrained(model_path)


    ds = load_dataset('json', data_files=data_files)

    # 查看特征
    column_names = list(ds["train"].features)
    print(column_names)

    # 批量处理
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        num_proc=10,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )


    # 这里我们取块长为 2048
    block_size = 2048

    # 批量处理
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=0,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size = 40000,
    )
    train_dataset = lm_datasets["train"]