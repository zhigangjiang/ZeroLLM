CUDA_VISIBLE_DEVICES=0,1

deepspeed train/sft_train_Qwen2.5-1.5B.py \
    --model_name_or_path /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B \
    --train_files /root/projects/happy-llm/ZeroLLM/autodl-tmp/dataset/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir  /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/sft_train_Qwen2.5-1.5B/output \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --warmup_steps 200 \
    --logging_dir /root/projects/happy-llm/ZeroLLM/autodl-tmp/model/sft_train_Qwen2.5-1.5B/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 900 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./train/ds_config_zero2.json \
    --report_to swanlab \
    # --evaluation_strategy  no \
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \