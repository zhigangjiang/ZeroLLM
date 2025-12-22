# 设置可见显卡
CUDA_VISIBLE_DEVICES=0

deepspeed train/pretrain_Qwen2.5-1.5B.py \
    --config_name /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B \
    --tokenizer_name /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B \
    --train_files /root/projects/happy-llm/ZeroLLM/autodl-tmp/dataset/seq_monkey_datawhale_small.jsonl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/pretrain_Qwen2.5-1.5B/output \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/pretrain_Qwen2.5-1.5B/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./train/ds_config_zero2.json \
    --report_to swanlab \
    # --evaluation_strategy  no \