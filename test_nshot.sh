#!/bin/bash

device=$1 #0/1/2/3
task=$2 #wsd/wsd_comp/wsd_bin/wsd_4shot_seed3/wsd_4shot_seed3_eos/wic_test/wic_test_eos/wic_test_4shot/wic_test_4shot_eos
model_id=$3 #llama_7b/llama_13b/mistral_7b/flan_t5_xl
model_type=$4 #base/chat
n_shot=$5 #0/1/2/3/...

run=${task}_${model_id}_${model_type}_${n_shot}shot
echo "Run : $run"

# Dataset related
data=$(dirname $BASH_SOURCE)/outputs/$run
wef_test=$(dirname $BASH_SOURCE)/data/WSD_Evaluation_Framework/Evaluation_Datasets

### base model
if [[ $model_id = 'llama_13b' ]]; then
   if [[ $model_type = 'chat' ]]; then
      base_model=/home/llm/models/llama2/llama-2-13b-chat-hf # twinkle 1
    #   base_model=/home/llm2/models/llama2/llama-2-13b-chat-hf # twinkle 2
   else
      base_model=/home/llm/models/llama2/llama-2-13b-hf # twinkle 1
    #   base_model=/home/llm2/models/llama2/llama-2-13b-hf # twinkle 2
   fi
elif [[ $model_id = 'llama_7b' ]]; then
   if [[ $model_type = 'chat' ]]; then
      base_model=/home/llm/models/llama2/llama-2-7b-chat-hf # twinkle 1
    #   base_model=/home/llm2/models/llama2/llama-2-7b-chat-hf # twinkle 2
   else
      base_model=/home/llm/models/llama2/llama-2-7b-hf # twinkle 1
    #   base_model=/home/llm2/models/llama2/llama-2-7b-hf # twinkle 2
   fi
elif [[ $model_id = 'mistral_7b' ]]; then
   base_model=/home/llm/models/Mistral-7B-v0.1 # twinkle 1
#    base_model=/home/llm2/models/Mistral-7B-v0.1 # twinkle 2
elif [[ $model_id = 'flan_t5_xl' ]]; then
   base_model=/home/llm/models/flan-t5-xl # twinkle 1
#    base_model=/home/llm2/models/flan-t5-xl # twinkle 2
fi

wsd_tasks="wsd wsd_comp wsd_bin wsd_4shot_seed3 wsd_4shot_seed3_eos"
wic_tasks="wic_test wic_test_eos wic_test_4shot wic_test_4shot_eos"

lm_eval \
    --model hf \
    --model_args pretrained=${base_model} \
    --tasks ${task} \
    --device cuda:${device} \
    --batch_size 4 \
    --num_fewshot ${n_shot} \
    --output_path ${data} \
    --log_samples

if [[ $wsd_tasks =~ $task ]]; then
   if [ $task = "wsd_bin" ]; then
      # # randomized
      # python scripts/parse_results_bin-wsd.py \
      #    --data_path $data \
      #    --output_path $data/ALL.txt \
      #    --randomize

      python scripts/parse_results_bin-wsd.py \
               --data_path $data \
               --output_path $data/ALL.txt
   else
      # # normalization
      # python scripts/parse_results_wsd.py \
      #     --data_path $data \
      #     --model_path $model_path

      # without normalization
      python scripts/parse_results_wsd.py \
         --data_path $data
   fi

   echo > $data/score.txt
   $java_path/java -cp $wef_test:. Scorer $wef_test/ALL/ALL.gold.key.txt $data/ALL.txt | tee -a $data/score.txt
fi
