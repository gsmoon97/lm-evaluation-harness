#!/bin/bash

task=wsd_gpt4
n_shot=$1 #0/1/2/3/...
model_name=gpt-4-0125-preview
run=${task}_${model_name}_${n_shot}shot
echo "Run : $run"

# Java related
java_path=/home/moongs/jvm/jdk-20.0.2/bin

# Dataset related
data=$(dirname $BASH_SOURCE)/outputs/$run
wef_test=$(dirname $BASH_SOURCE)/data/WSD_Evaluation_Framework/Evaluation_Datasets

# export OPENAI_API_KEY=...
lm_eval \
    --model openai-chat-completions \
    --model_args model=${model_name} \
    --gen_kwargs max_tokens=1024,temperature=0 \
    --tasks ${task} \
    --num_fewshot $n_shot \
    --output_path $data \
    --log_samples

# python scripts/parse_results_wsd.py \
#     --data_path $data

# echo > $data/score.txt
# $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data/ALL.txt | tee -a $data/score.txt
