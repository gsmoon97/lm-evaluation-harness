#!/bin/bash

device=$1 #0/1/2/3
task=$2 #wsd/wsd_eos/wsd_comp/wsd_4shot_seed3/wsd_5shot_seed3/wsd_bin/wsd_4shot_seed3_eos/wsd_chat
model_id=$3 #llama2/mistral/flan-t5
if [[ $model_id = "llama2" ]]; then
    model_size="13b" #13b/7b
elif [[ $model_id = "mistral" ]]; then
    model_size="7b" #7b
elif [[ $model_id = "flan-t5" ]]; then
    model_size="xl" #xl
fi
model_type=$4 #base/chat
n_shot=$5 #0/1/2/3/...
batch_size=$6 #"auto:4"
quantization=False #True/False
if [[ $quantization = "True" ]]; then
    run=${task}_${model_id}_${model_size}_${model_type}_${n_shot}shot_quantization
elif [[ $quantization = "False" ]]; then
    run=${task}_${model_id}_${model_size}_${model_type}_${n_shot}shot
else
    echo "!!!Quantization is not set!!!"
    exit 0
fi
echo "Run : $run"

# Quantization related
use_4bit=True
bnb_4bit_quant_type=nf4
bnb_4bit_compute_dtype=float16
use_nested_quant=False

# Java related
java_path=/home/moongs/jvm/jdk-20.0.2/bin

# Dataset related
data=$(dirname $BASH_SOURCE)/outputs/$run
wef_test=$(dirname $BASH_SOURCE)/data/WSD_Evaluation_Framework/Evaluation_Datasets

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

if [[ $quantization = "True" ]]; then
    lm_eval \
        --model hf \
        --model_args pretrained=${model_path},load_in_4bit=${use_4bit},bnb_4bit_quant_type=${bnb_4bit_quant_type},bnb_4bit_compute_dtype=${bnb_4bit_compute_dtype},bnb_4bit_use_double_quant=${use_nested_quant} \
        --tasks ${task} \
        --device cuda:$device \
        --batch_size ${batch_size} \
        --num_fewshot $n_shot \
        --output_path $data \
        --log_samples
else
    lm_eval \
        --model hf \
        --model_args pretrained=${model_path} \
        --tasks ${task} \
        --device cuda:$device \
        --batch_size ${batch_size} \
        --num_fewshot $n_shot \
        --output_path $data \
        --log_samples
fi

if [ $task = "wsd_bin" ]; then
    # not randomized
    python scripts/parse_results_bin-wsd.py \
        --data_path $data \
        --output_path $data/ALL.txt

    echo > $data/score.txt
    $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data/ALL.txt | tee -a $data/score.txt
    
    # randomized
    python scripts/parse_results_bin-wsd.py \
        --data_path $data \
        --output_path $data/ALL_rand.txt \
        --randomize

    echo > $data/score_rand.txt
    $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data/ALL_rand.txt | tee -a $data/score_rand.txt
else
    python scripts/parse_results_wsd.py \
        --data_path $data

    echo > $data/score.txt
    $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data/ALL.txt | tee -a $data/score.txt
fi
