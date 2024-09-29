# Copyright 2024 Re2 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
import json 
import argparse
from collections import defaultdict
import random 
from utils.scorer import dataset2scorer
from utils.answer_parser import dataset2parser
from collections import Counter
random.seed(1234)


def main(args):
    # Read jsonl files 
    eval_file = args.eval_file
    eval_datas = [json.loads(line) for line in open(eval_file, 'r').readlines()]
    total_num = len(eval_datas)
    correct_num_dict = {act: 0 for act in args.acts}
    null_num_dict = {act: 0 for act in args.acts}
    no_boxed_num_dict = {act: 0 for act in args.acts}
    score_func = dataset2scorer[args.dataset]

    for eval_data in eval_datas:
        for act in args.acts:
            extract_answer_func = dataset2parser[args.dataset]
            gens = eval_data[act]['generation']
            pred = None
            if gens is not None:
                results = []
                for gen in gens:
                    answer = extract_answer_func(gen, eval_data)
                    results.append(answer)
                if len(gens) != 0 and "boxed" not in gens[0]:
                    no_boxed_num_dict[act] += 1

            counter = Counter(results)
            pred = counter.most_common(1)[0][0]                

            if pred is None:
                null_num_dict[act] += 1
                continue

            score = score_func(pred, eval_data)
            correct_num_dict[act] += score
            # Write the answer and score 
            eval_data[act]['answer'] = pred
            eval_data[act]['score'] = score
        
    # Write the results to jsonl file
    output_file = eval_file.replace('.jsonl', f'_eval.jsonl')
    print("Write the results to {}".format(output_file))
    with open(output_file, 'w') as f:
        for eval_data in eval_datas:
            f.write(json.dumps(eval_data) + '\n')

    print(f"Total num: {total_num}")
    for act in args.acts:
        correct_num = correct_num_dict[act]
        null_num = null_num_dict[act]
        print(f'act: {act}, correct_num: {correct_num}, null_num: {null_num}, accuracy: {(correct_num / total_num):.4f}, null_rate: {(null_num / total_num):.2f}, no_boxed_num: {no_boxed_num_dict[act]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default='results/gsm.jsonl')
    parser.add_argument('--acts', nargs="+", default=["vanilla", "cot", "ps", "pal"])
    parser.add_argument("--dataset", type=str, default="gsm")
    parser.add_argument("--re_extract", action="store_true", help="re-extract the answer from the generation")
    args = parser.parse_args()
    main(args)
