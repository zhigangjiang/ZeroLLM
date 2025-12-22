# ZeroLLM
从0搭建LLM(基于LLaMA2)

## 数据预处理
### 下载数据
下载预训练数据

Token。
``` bash
bash download/download_pretrain_dataset.sh
```
- 出门问问序列猴子开源数据集：出门问问序列猴子通用文本数据集由来自网页、百科、博客、问答、开源代码、书籍、报刊、专利、教材、考题等多种公开可获取的数据进行汇总清洗之后而形成的大语言模型预训练语料。总量大概在 10B Token。

- 包含13000000条文本数据




下载SFT训练数据，监督微调（Supervised Fine-Tuning，SFT
``` bash
bash download/download_sft_dataset.sh
```
- BelleGroup：350万条中文对话数据集，包含了人机对话、人人对话、人物对话等多种对话数据，可以用于训练对话生成模型。

- 包含3606402条多轮对话数据

### 处理数据
处理预训练数据
``` bash
python dataset/pre_process/deal_pretrain_dataset.py
```
按chunk_size=512拆分，处理后有28998989条文本数据

处理SFT训练数据
``` bash
python dataset/pre_process/deal_sft_dataset.py
```
处理为标注格式
## 训练Tokenizer
> 可以直接使用训练好的Tokenizer，位于`./tokenizer_k`，跳过此步骤

``` bash
python train/train_tokenizer.py
```
## 训练Base模型
``` bash
nohup python ./train/pretrain.py --use_swanlab &
```
> 在单卡NVIDIA RTX PRO 6000(96GB) 上，batchsize可以设置到128，实验默认使用64，且为快速验证仅用前1000万条数据（在`dataset/pretrain_dataset.py#17`修改）

![pretrain_swanlab.png](docs/images/pretrain_swanlab.png)

训练好的模型：[🤗model地址](https://huggingface.co/zhigangjiang/ZeroLLM/resolve/main/base_model_215M/pretrain_1024_18_6144.pth)

推理测试
``` bash
(llm) root@autodl-container-lhy2360kfm-b38ddd38:~/projects/happy-llm/ZeroLLM# /root/miniconda3/envs/llm/bin/python /root/projects/happy-llm/ZeroLLM/inference/model_sample.py
Model has 215.127 M parameters.

Sample 1:
<|im_start|>北京大学是中国最高学府,也是教育部直属的最高学府,北京大学在教育部此次排名中排位第2,师资力量、研究水平、办学层次等均居全国高校前列。2019年,北京大学招收38个本科专业,招收35个硕士研究生,招收7个博士研究生,招收17个博士后流动站。2020年,北京大学招收37个博士研究生
--------------------

Sample 2:
<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院副教授黄河认为,地下水将成为石油和天然气行业重要的战略资源。
据了解,中国矿业大学(北京)地球科学与测绘工程学院的师生们有幸参与“矿业2030”计划,在黄河水利委员会、中国石油和全国石油公司、中国石油大学(北京)、中国科学院地球科学研究所等单位的有关部门和单位的大力支持下,成功开展了
--------------------
```
## 训练SFT模型


## 基于HF训练
