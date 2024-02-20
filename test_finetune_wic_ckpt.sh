#!/bin/bash

device=$1 #0/1/2/3
task=$2 #wic_test/wic_test_eos
model_id=$3 #llama2/mistral/t5
if [[ $model_id = "llama2" ]]; then
    model_size="13b" #13b/7b
elif [[ $model_id = "mistral" ]]; then
    model_size="7b" #7b
elif [[ $model_id = "t5" ]]; then
    model_size="xl" #xl
fi
model_type=base
batch_size="auto:4" #"auto:4"/2/4/
run=${task}_${model_id}_${model_size}_${model_type}
suffix_root=_${model_size}_${model_type}

#hyperparameters
learning_rate=$4 # 2e-4/1e-5/5-e5
if [[ -n "${learning_rate}" ]]; then
    run=${run}_lr_${learning_rate}
    suffix_root=${suffix_root}_lr_${learning_rate}
fi
echo "Run : $run"

# Quantization related
use_4bit=True
bnb_4bit_quant_type=nf4
bnb_4bit_compute_dtype=float16
use_nested_quant=False

# Dataset related
data=$(dirname $BASH_SOURCE)/outputs/$run

if [[ $model_id = "llama2" ]]; then
    model_path=/home/llm/models/llama2/llama-2-${model_size}-hf # twinkle1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments_wic/V1_hf/result_llama${suffix_root}
elif [[ $model_id = "mistral" ]]; then
    model_path=/home/llm/models/Mistral-7B-v0.1 # twinkle1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments_wic/V1_hf/result_mistral${suffix_root}
elif [[ $model_id = 't5' ]]; then
    model_path=/home/llm/models/flan-t5-xl # twinkle 1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments_wic/V1_hf/result_flan_t5${suffix_root}
fi

# find the best checkpoint
CHECK_POINT_VALUES=(169 339 507) 
for CHECK_POINT in "${CHECK_POINT_VALUES[@]}"
do
    echo "Running script with CHECK_POINT=$CHECK_POINT"

    peft_model_path=${peft_model_path_root}/checkpoint-${CHECK_POINT}
    data_ckpt=$data/ckpt-${CHECK_POINT}

    echo "peft_model_path : ${peft_model_path}"
    echo "data_ckpt : ${data_ckpt}"
    
    lm_eval \
        --model hf \
        --model_args pretrained=${model_path},peft=${peft_model_path},load_in_4bit=${use_4bit},bnb_4bit_quant_type=${bnb_4bit_quant_type},bnb_4bit_compute_dtype=${bnb_4bit_compute_dtype},bnb_4bit_use_double_quant=${use_nested_quant} \
        --tasks $task \
        --device cuda:$device \
        --batch_size $batch_size \
        --output_path $data \
        --log_samples
done
