#!/bin/bash

device=$1 #0/1/2/3
# task=super-glue-lm-eval-v1
# task=boolq
# task=cb
# task=copa
# task=multirc
# task=record
# task=rte
# task=wic
# task=wsc
task=$2 #wic_test/wic_test_4shot/wic_test_4shot_eos
model_id=$3 #llama2/mistral/flan-t5
if [[ $model_id = "llama2" ]]; then
    model_size="13b" #13b/7b
elif [[ $model_id = "mistral" ]]; then
    model_size="7b" #7b
elif [[ $model_id = "flan-t5" ]]; then
    model_size="xl" #xl
fi
model_type=base
n_shot=$4 #0/1/2/3/...
batch_size=auto:4 #"auto:4"/2/4/
run=${task}_${model_id}_${model_size}_${model_type}_${n_shot}shot
echo "Run : $run"

# Dataset related
data=$(dirname $BASH_SOURCE)/outputs/$run

if [[ $model_id = "llama2" ]]; then
    if [[ $model_type = "chat" ]]; then
        model_path=/home/llm/models/llama2/llama-2-${model_size}-${model_type}-hf
    else
        model_path=/home/llm/models/llama2/llama-2-${model_size}-hf
    fi
elif [[ $model_id = "mistral" ]]; then
    model_path=/home/llm/models/Mistral-7B-v0.1
elif [[ $model_id = "flan-t5" ]]; then
    model_path=/home/llm/models/flan-t5-xl
fi

lm_eval \
    --model hf \
    --model_args pretrained=${model_path} \
    --tasks ${task} \
    --device cuda:${device} \
    --batch_size ${batch_size} \
    --num_fewshot ${n_shot} \
    --output_path ${data} \
    --log_samples
