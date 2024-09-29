# Copyright 2023 PAL Authors. All rights reserved.
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

import copy
import json
import argparse
import tqdm
import os
import openai 
from pal import interface
from prompter import dataset2prompter as dataset2prompter
from utils.multithread_call import MultithreadCall
import re
from functools import partial
import datetime
from utils.answer_parser import dataset2parser
from utils.scorer import dataset2scorer
from pal.core.runtime import dataset2runtime


def run(example, act="vanilla"):
    prompter = dataset2prompter[args.dataset](args.dataset) # TDOO: refract this
    sys_message = prompter.get_sys_message(act)
    prompt = prompter.build_prompt(example, act, read_times=args.read_times)
    if args.verbose:
        print("SYS_MESSAGE:")
        print(sys_message)
        print("PROMPT:")
        print(prompt)
    if act != "pal":
        itf = interface.TextChatInterface(
            stop=None,
            model=args.model,
            system_message=sys_message,
            extract_answer=dataset2parser[args.dataset],
            verbose=args.verbose,
        )
        try:
            ans = itf.run(
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                majority_at=args.majority_at,
                example=example,
                api_source=args.api_source,
            )
            score = dataset2scorer[args.dataset](ans, example)  
        except Exception as e:
            print("ERROR:") 
            print(e)
            score = 0
            if "ans" not in locals():
                ans = None
            # traceback_info = traceback.format_exc()
            # print(traceback_info)
        gen = itf.history[-1] if len(itf.history) > 0 else None
    else:
        itf = interface.ProgramInterface(
            stop=None,
            get_answer_expr='solution()',
            model=args.model,
            verbose=args.verbose,
            system_message=sys_message,
            runtime=dataset2runtime[args.dataset],
            extract_answer=dataset2parser[args.dataset],
        )
        try:
            ans = itf.run(
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                majority_at=args.majority_at,
                example=example,
                api_source=args.api_source,
            )
            score = dataset2scorer[args.dataset](ans, example)
        except Exception as e:
            print("ERROR:")
            print(e)
            score = 0
            if "ans" not in locals():
                ans = None
        gen = itf.history[-1] if len(itf.history) > 0 else None
    
    itf.clear_history()
    if args.verbose:
        print("Extracted answer:")
        print(ans)
        print("SCORE:")
        print(score)

    return gen, ans, score


def multi_acts_run(example, acts=["vanilla", "cot", "ps", "pal"], regen_null=False):
    for act in acts:
        if regen_null:
            # Regerate the sample with null generation.
            if example[act]["generation"] is None or example[act]["answer"] is None:
                gen, ans, score = run(example=example, act=act)
            else:
                continue 
        else:
            gen, ans, score = run(example=example, act=act)

        example[act] = {
            "generation": gen,
            "answer": ans,
            "score": score
        }
    return example
  

def main(args):
    if args.file_name is None:
        args.file_name = args.dataset
    data_path = os.path.join(args.data_dir, args.file_name + ".jsonl")
    examples = list(map(json.loads, open(data_path)))

    if args.dataset in ["svamp", "multiarith", "mawpssingleeq", "mawpsaddsub"]: 
        args.dataset = "gsm"    # Use the prompt of gsm for math dataset
    elif args.dataset in ["arc_easy", "arc_challenge"]:
        args.dataset = "commonsenseqa"  # Use the prompt of commonsenseqa for arc dataset. They are the same in format.
    elif args.dataset in ['prealgebra', 'algebra', 'number_theory', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'precalculus']:
        args.dataset = "MATH"
    if args.samples_num:
        examples = examples[args.start_idx:args.start_idx + args.samples_num]

    if args.debug:
        examples = examples[:4]
    
    # Add id field to each example.
    for i, example in enumerate(examples):
        example["id"] = i

    debug_suffix = "_debug" if args.debug else ""
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.append:
        output_path = args.result_path
        assert output_path
        processed_examples = [json.loads(line) for line in open(output_path, 'r').readlines()]
        processed_ids = list(map(lambda x: x["id"], processed_examples))
        examples = [example for idx, example in enumerate(examples) if idx not in processed_ids]
        print(f"Append to {output_path} with {len(examples)} examples.")
    else:
        if args.regen_null: 
            assert args.result_path
            examples = list(map(json.loads, open(args.result_path, 'r').readlines()))
            basename = ".".join(os.path.basename(args.result_path).split('.')[:-1])
            output_path = f'results/{basename}_regen.jsonl'
        else:
            output_path = f'results/{args.file_name}_{args.model}{debug_suffix}_topp{args.top_p}_temp{args.temperature}_majority{args.majority_at}_readtimes{args.read_times}_{run_time}.jsonl'
            assert not os.path.exists(output_path)
    print("DATA_PATH:", data_path)
    print("OUTPUT_PATH:", output_path)
    
    if not args.multithread:
        # Generation
        with open(output_path, 'a' if args.append else 'w') as f:
            for i, json_data in enumerate(tqdm.tqdm(examples)):
                json_data = multi_acts_run(example=json_data, acts=args.acts, regen_null=args.regen_null)
                f.write(json.dumps(json_data) + '\n')
                f.flush()
    else:        
        caller = MultithreadCall(
            examples=examples,
            fn=partial(multi_acts_run, acts=args.acts, regen_null=args.regen_null),
            output_file=output_path,
            num_threads=args.num_threads,
            append=args.append
        )
        caller.run()
    
    print(f"Results are saved in {output_path}")

    if args.regen_null:
        # Backup the original file and rename the new file.
        os.system(f"cp {args.result_path} {args.result_path}.bak")
        os.system(f"mv {output_path} {args.result_path}")
        print(f"Backup the original file and rename the new file to {args.result_path}")
        output_path = args.result_path

    # Test 
    print("Evaluating the results...")
    print(f"python eval.py --acts {' '.join(args.acts)} --eval_file {output_path} --dataset {args.dataset}")
    os.system(f"python eval.py --acts {' '.join(args.acts)} --eval_file {output_path} --dataset {args.dataset}")
    print("Evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--data_dir', default="../data/reasoning_tasks", type=str)
    parser.add_argument('--file_name', default=None, type=str, help="The file name of the dataset. If not specified, it will be the same as the dataset name.")
    parser.add_argument('--dataset', default='gsm', type=str)
    parser.add_argument('--result_path', default=None, type=str, help="The path to store the results. You can do not specify it and the results will be stored in the results folder.")
    parser.add_argument('--model', default='gpt-3.5-turbo-0613', type=str)
    parser.add_argument('--majority_at', default=1, type=int, help="The number of majority vote for the answer")
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--top_p', default=1.0, type=float)
    parser.add_argument('--max_tokens', default=1024, type=int)
    parser.add_argument('--multithread', action='store_true', help="Whether to use multithread to call the API")
    parser.add_argument('--num_threads', default=40, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--acts', nargs="+", default=['vanilla', 'cot', 'ps', 'pal'], help="The list of acts for reasoning")
    parser.add_argument('--regen_null', action='store_true', help="Whether to regenerate the sample with null generation")
    parser.add_argument('--samples_num', default=None, type=int, help="The number of samples to generate")
    parser.add_argument('--start_idx', default=0, type=int, help="The start index of the samples to generate")
    parser.add_argument('--read_times', default=1, type=int, help="The times of repetition of the question")
    parser.add_argument('--api_source', choices=["openai", "azure"], default="openai", help="The source of the API")
    
    args = parser.parse_args()
    # Print args gracefully
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
    openai.api_version = os.getenv("OPENAI_API_VERSION", None)
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    main(args)