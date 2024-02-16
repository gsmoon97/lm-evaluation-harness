"""
Usage:
   python parse_results_bin-wsd.py --data_path <path_to_jsonl_file>
"""
import argparse
import json
import math
import os

from collections import OrderedDict

# ins_data is a dictionary of `sense_key`: `ins_data`.
def choose_answer(ins_data):
    def choose_highest(data):
        max_prob = -math.inf
        argmax_prob = None
        for sense_key, sense_data in data.items():
            if sense_data['yes_prob'] > max_prob:
                max_prob = sense_data['yes_prob']
                argmax_prob = sense_key
        assert argmax_prob is not None, "No choices!"

        return {
            'sense_key': argmax_prob,
            # 'answer': data[argmax_prob]['answer'] # for approx score calculation
        }

    yes_anwers = {} # a subset dict where yes_bool == True
    for sense_key, sense_data in ins_data.items():
        if sense_data['yes_bool']:
            yes_anwers[sense_key] = sense_data
    
    if len(yes_anwers) == 1:
        sense_key = next(iter(yes_anwers)) # get the only key
        return {
            'sense_key': sense_key,
            # 'answer': yes_anwers[sense_key]['answer'] # for approx score calculation
        }
    elif len(yes_anwers) > 1:
        return choose_highest(yes_anwers)
    else:
        return choose_highest(ins_data)


def main(args):
    jsonl_files = [name for name in os.listdir(args.data_path) if name.endswith('.jsonl')]
    assert len(jsonl_files) <= 1, f'Found more than one .jsonl file in {args.data_path}!'
    with open(os.path.join(args.data_path, jsonl_files[0]), "r") as f:
        data = json.load(f)
    
    raw_answers = OrderedDict()
    for doc in data:
        sense_key = doc['doc']['sense_key']
        id = doc['doc']['id']
        if id not in raw_answers:
            raw_answers[id] = {}
        assert sense_key not in raw_answers[id], "duplicate sense key answers!"

        arguments = [a[-1].lower().strip() for a  in doc['arguments']]
        resps = [r[0] for r in doc['resps']]
        assert len(arguments) == len(resps)
        
        # 1. Considered as 'yes' if the bool for argument yes is true
        # 2. If prob of 'yes' > 'no', also considered as 'yes'
        yes_bool = False
        yes_prob = None
        max_prob = -math.inf
        argmax_prob = ''
        for a, (p, b) in zip(arguments, resps):
            if a == "yes":
                yes_prob = p
                if b:
                    yes_bool = True
            if p > max_prob:
                max_prob = p
                argmax_prob = a
        if argmax_prob == "yes":
            yes_bool = True
        assert yes_prob is not None, f"No 'yes' argument found for instance {id}!"

        raw_answers[id][sense_key] = {
            'yes_bool': yes_bool,
            'yes_prob': yes_prob,
            # 'answer': doc['doc']['answer'].strip().lower(),
        }
    
    final_answers = OrderedDict()
    tp = 0
    fp = 0
    for id, ins_data in raw_answers.items():
        highest = choose_answer(ins_data)
        final_answers[id] = highest['sense_key']
        if 'answer' in highest:
            if highest['answer'] == 'yes':
                tp += 1
            else:
                fp += 1
    
    if tp + fp > 0:
        print('Precision:', float(tp)/(tp+fp))
    output_path = args.output_path or os.path.join(os.path.realpath(__file__), 'ALL.txt')
    with open(output_path, 'w', encoding='utf-8') as out:
        for k, v in final_answers.items():
            out.write(f'{k} {v}\n')
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help='directory to the LM-Eval output files.')
    parser.add_argument("--output_path", type=str, default=None, help='file path for the parsed file.')
    args = parser.parse_args()
    main(args)
