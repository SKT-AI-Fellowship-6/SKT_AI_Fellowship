#!/bin/bash
set -xe

mkdir checkpoint

# Ferret_13b_train.sh 를 바탕으로 수정함
data_path=('/home/sunmin/dataset/json_files')
image_folder=('/home/sunmin/dataset/image_files')
data_multiple(1)

# convert array to string
data_path="${data_path[@]}"
image_folder="${image_folder[@]}"

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-13b-v1-3"
################## VICUNA ##################

torchrun --nnodes 1 --nproc_per_node 8 /home/sunmin/ml-ferret/ferretui/ferretui/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./model/$MODEL_VERSION \
    --data_path $data_path \
    --data_multiple 1 \
    --image_aspect_ratio anyres \
    --image_folder $image_folder \
    --is_multimodal False \
    --lazy_preprocess True \
    --resized_image_h 336 \
    --resized_image_w 336 \
    --point_input_sample segment_mask-center \
    --refer_previous_point False \
    --bf16 True \
    --dataloader_num_workers 8 \
    --evaluation_strategy no \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --output_dir ./checkpoint/ferret_ui_finetune \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --report_to wandb \
    --save_steps 50000 \
    --save_strategy steps \
    --save_total_limit 1 \
    --tf32 True \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --freeze_backbone False \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type spatial \
    --mm_use_im_patch_token False \
    --mm_use_im_start_end False \
    --mm_vision_select_feature patch \
    --mm_vision_select_layer -2 \
    --tune_mm_mlp_adapter ./model/llava-336px-pretrain-$MODEL_VERSION/mm_projector.bin \
    --version ferret_gemma_instruct \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --add_region_feature True \
    --region_geo_sampler True \
    --sampler_pooler_mode max \
    --use_safetensors True \
    --model_max_length 4096
