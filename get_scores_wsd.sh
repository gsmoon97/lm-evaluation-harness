#!/bin/bash

# Java related
java_path=/home/moongs/jvm/jdk-20.0.2/bin

# Dataset related
run=$1
data=$(dirname $BASH_SOURCE)/outputs/$run
wef_test=$(dirname $BASH_SOURCE)/data/WSD_Evaluation_Framework/Evaluation_Datasets

python scripts/get_scores_wsd.py --data_path ${data}

echo > $data/score_splits.txt

test_dataset_splits=(semeval2007 semeval2013 semeval2015 senseval2 senseval3)
for tds in "${test_dataset_splits[@]}"
do
    echo ${tds} >> $data/score_splits.txt
    ${java_path}/java -cp ${wef_test}:. Scorer ${wef_test}/${tds}/${tds}.gold.key.txt ${data}/${tds}.txt | tee -a ${data}/score_splits.txt
done
