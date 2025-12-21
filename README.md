# ZeroLLM
从0搭建LLM(基于LLaMA2)

## 训练Tokenizer
下载训练数据

``` bash
bash ./data/download_dataset.sh
```

处理数据

``` py
python ./data/deal_dataset.py
```