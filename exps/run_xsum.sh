#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

# conda activate adapter
# wandb env variables
# export WANDB_PROJECT=xsum
# export WANDB_WATCH="false"

DATE=`date +%Y%m%d`
dataset="xsum"

# ----- MAM adapter -----
attn_mode="prefix"
attn_option="concat"
attn_composition="add"
attn_bn=30  # attn bottleneck dim

ffn_mode="adapter"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"
ffn_adapter_scalar="4"
ffn_bn=512 # ffn bottleneck dim

# ----- prefix tuning baseline -----
# attn_mode="prefix"
# attn_option="concat"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- Houlsby Adapter -----
# attn_mode="adapter"
# attn_option="sequential"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="sequential"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=200 # ffn bottleneck dim


# ----- FFN Scaled Parallel Adapter -----
# attn_mode="none"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- Prompt Tuning -----
# attn_mode="prompt_tuning"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- bitfit -----
# attn_mode="bitfit"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="none"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim

# ----- lora -----
# attn_mode="lora"
# attn_option="none"
# attn_composition="add"
# attn_bn=16

# # set ffn_mode to be 'lora' to use
# # lora at ffn as well

# ffn_mode="none"
# ffn_option="none"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=16

# lora_alpha=32
# lora_dropout=0.1
# lora_init="lora"


# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi


# set to 1 for debug mode which only
# uses 1600 training examples
debug=0

# set to "wandb" to use weights & bias
report_to="none"

label_smoothing_factor=0.1
weight_decay=0.01
max_grad_norm=0.1
max_steps=100000
num_train_epochs=30
warmup_updates=0
lr=5e-5
lr_scheduler_type="polynomial"
bsz=21
gradient_steps=3

metric=rouge2
unfreeze='ef_'
max_eval_samples=1600
logging_steps=100
eval_strategy="steps"
# eval_strategy="steps"
save_steps=3000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1
    max_train_samples=2000
    bsz=24
    gradient_steps=2
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples}"
    debug_str=".debug"
fi


exp_name=xsum.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${attn_bn}.fbn${ffn_bn}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}.fs_${ffn_adapter_scalar}
exp_name+=.unfrz_${unfreeze}.ms${max_steps}.ls${label_smoothing_factor}
exp_name+=.warm${warmup_updates}.wd${weight_decay}${debug_str}

SAVE=checkpoints/${dataset}/mam/${exp_name}

rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock

/home/MLTL/yiran-wang/miniconda3/envs/tuning/bin/python3 -u examples/pytorch/summarization/run_summarization.py \
    --dataset_name 'xsum' \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --unfreeze_params ${unfreeze} \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 60 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams 6 \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --label_smoothing_factor ${label_smoothing_factor} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${dataset}.mam.${exp_name} \
    --overwrite_output_dir "True" \
    --disable_tqdm "False" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
    # --predict_with_generate
    # --metric_for_best_model ${metric} \
    # --greater_is_better "True" \

#rm -rf ${SAVE}/pytorch_model.bin
