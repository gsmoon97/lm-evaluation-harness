#!/bin/bash

device=$1 #0/1/2/3
task=$2 #wsd/wsd_comp/wsd_bin
model_id=$3 #llama2/mistral/t5
if [[ $model_id = "llama2" ]]; then
    model_size="13b" #13b/7b
elif [[ $model_id = "mistral" ]]; then
    model_size="7b" #7b
elif [[ $model_id = "t5" ]]; then
    model_size="xl" #xl
fi
model_type=base
sample=$4 #10k/20k/
downsample=True #True/False
if [[ -n "${sample}" ]]; then
    run=${task}_${model_id}_${model_size}_${model_type}_${sample}
    suffix_root=_${model_size}_${model_type}_${sample}
else
    # fully finetuned
    run=${task}_${model_id}_${model_size}_${model_type}
    suffix_root=_${model_size}_${model_type}
fi
if [ $downsample = "True" ]; then
   run=${run}_downsampled
   suffix_root=${suffix_root}_downsampled
fi
# learning_rate="1e-5" # 2e-4/1e-5
# run=${run}_lr_${learning_rate}
# suffix_root=${suffix_root}_lr_${learning_rate}
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

# Model related
# if [[ $model_type = "chat" ]]; then
#     model_path=/home/llm/models/llama2/llama-2-${model_size}-${model_type}-hf # twinkle1
# else
#     model_path=/home/llm/models/llama2/llama-2-${model_size}-hf # twinkle1
# fi
if [[ $task = "wsd" ]]; then
    style=MCQ_NUM
elif [[ $task = "wsd_comp" ]]; then
    style=COMP
elif [[ $task = "wsd_bin" ]]; then
    style=BIN
fi
if [[ $model_id = "llama2" ]]; then
    model_path=/home/llm/models/llama2/llama-2-${model_size}-hf # twinkle1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments/${style}_v2/result_llama${suffix_root}
    # peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments/${style}_v2/result_llama${suffix_root}_mask # temporary fix for correct path
elif [[ $model_id = "mistral" ]]; then
    model_path=/home/llm/models/Mistral-7B-v0.1 # twinkle1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments/${style}_v2/result_mistral${suffix_root}
elif [[ $model_id = 't5' ]]; then
    model_path=/home/llm/models/flan-t5-xl # twinkle 1
    peft_model_path_root=/home/moongs/workspace/lm-evaluation-harness/experiments/${style}_v2/result_flan_t5${suffix_root}
fi

# find the best checkpoint
CHECK_POINT_VALUES=(1028 2056 3084)
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
        --batch_size auto:4 \
        --output_path $data_ckpt \
        --log_samples

    if [ $task = "wsd_bin" ]; then
        # not randomized
        python scripts/parse_results_bin-wsd.py \
            --data_path $data_ckpt \
            --output_path $data_ckpt/ALL.txt

        echo > $data_ckpt/score.txt
        $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data_ckpt/ALL.txt | tee -a $data_ckpt/score.txt
        
        # randomized
        python scripts/parse_results_bin-wsd.py \
            --data_path $data_ckpt \
            --output_path $data_ckpt/ALL_rand.txt \
            --randomize

        echo > $data_ckpt/score_rand.txt
        $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data_ckpt/ALL_rand.txt | tee -a $data_ckpt/score_rand.txt
    else
        # # normalization
        # python scripts/parse_results_wsd.py \
        #     --data_path $data_ckpt \
        #     --model_path $model_path

        # without normalization
        python scripts/parse_results_wsd.py \
            --data_path $data_ckpt

        echo > $data_ckpt/score.txt
        $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data_ckpt/ALL.txt | tee -a $data_ckpt/score.txt
    fi
done
