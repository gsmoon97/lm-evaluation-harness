"""
Usage:
   python get_scores_wsd.py --data_path <path_to_score_files>
"""
import argparse
import json
import math
import os
import random
import warnings

import transformers


def main(args):
    with open(os.path.join(args.data_path, "ALL.txt"), "r") as f:
        data = f.readlines()
    
    semeval2007 = []
    semeval2013 = []
    semeval2015 = []
    senseval2 = []
    senseval3 = []

    for d in data:
        # print(d)
        line = d.split(".",1)[1]
        # print(line)
        instance_id = d.split(" ")[0]
        # print(instance_id)
        test_dataset_split = instance_id.split(".")[0]
        # print(test_dataset_split)
        if test_dataset_split == "semeval2007":
            semeval2007.append(line)
        elif test_dataset_split == "semeval2013":
            semeval2013.append(line)
        elif test_dataset_split == "semeval2015":
            semeval2015.append(line)
        elif test_dataset_split == "senseval2":
            senseval2.append(line)
        elif test_dataset_split == "senseval3":
            senseval3.append(line)
        else:
            raise Warning(f"Invalid test_dataset_split '{test_dataset_split}' found!")
        
    test_dataset_split_dict = {
        "semeval2007" : semeval2007,
        "semeval2013" : semeval2013,
        "semeval2015" : semeval2015,
        "senseval2" : senseval2,
        "senseval3" : senseval3,
    }
    
    for test_dataset_split in ["semeval2007", "semeval2013", "semeval2015", "senseval2", "senseval3"]:
        with open(os.path.join(args.data_path, f"{test_dataset_split}.txt"), "w") as f:
            pred_lines = test_dataset_split_dict[test_dataset_split]
            pred_lines[-1].strip('\n')
            f.writelines(pred_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
