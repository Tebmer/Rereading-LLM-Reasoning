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



from typing import Union, Any
from math import isclose
from utils.answer_parser import normalize_answer
from utils.util import last_boxed_only_string, remove_boxed, _strip_string

def scorer_number(pred, example):
    target = example["target"]
    if pred is None: 
        return 0
    try:
        pred, target = float(pred), float(target)
        score = 1 if abs(pred - target) < 1e-3 else 0
    except:
        score = 0
    return score


def scorer_string(pred, example):
    target = example["target"]
    if pred is None: 
        return 0
    score = 1 if pred.lower() == target.lower() else 0
    return score


def scoere_date_understanding(pred, example):
    target_scores = example["target_scores"]
    if pred is None or pred not in target_scores:
        return 0
    score = target_scores[pred]
    return score


def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision


def finqa_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = False,
                is_close: float = False) -> bool:
    if prediction is None:
        return False
    elif type(prediction) == bool:
        # bool questions
        if prediction:
            return reference == 'yes'
        else:
            return reference == 'no'
    elif type(reference) == str or type(prediction) == str:
        # string questions
        return prediction == reference
    else:
        # number questions
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_close:
                    if isclose(item, prediction, rel_tol=0.001):
                        return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False


def scorer_tabmwp(pred, example):
    # Process ground truth ansewr according to TabMWP.
    if pred is None:
        return 0
    try:
        pred = float(pred)
    except:
        pred = str(pred)
    gt_ans = example['answer']
    include_percentage = False
    if example['ans_type'] in ['integer_number', 'decimal_number']:
        if '/' in gt_ans:
            gt_ans = int(gt_ans.split('/')[0]) / int(gt_ans.split('/')[1])
        elif ',' in gt_ans:
            gt_ans = float(gt_ans.replace(',', ''))
        elif '%' in gt_ans:
            gt_ans = float(gt_ans.split('%')[0]) / 100
            include_percentage = True
            # Also process the prediction.
            try:
                pred = float(pred.split('%')[0]) / 100
            except:
                pass
        else:
            gt_ans = float(gt_ans)
    elif example['ans_type'].endswith('_text'):
        gt_ans = str(gt_ans)
    else:
        raise ValueError(example['ans_type'])
    if finqa_equal(pred, gt_ans, include_percentage=include_percentage, is_close=True):
        score = 1
    else:
        score = 0
    return score


def scorer_tabwmp_official(pred, example):
    answer = example["answer"]
    if pred is None:
        return 0
    pred = normalize_answer(pred, example['unit'])
    answer = normalize_answer(answer, example['unit'])
    if answer.lower() == pred.lower():
        score = 1
    else:
        score = 0
    return score


def scorer_commonsenseqa(pred, example):
    target = example["answerKey"]
    if pred is None: 
        return 0
    score = 1 if pred.lower() == target.lower() else 0
    return score


def scorer_aqua(pred, example):
    target = example["correct"]
    if pred is None: 
        return 0
    score = 1 if pred.lower() == target.lower() else 0
    return score
    

def scorer_MATH(pred, example):
    # Refer to "https://github.com/hendrycks/math/blob/main/modeling/dataset/util.py"
    answer = remove_boxed(last_boxed_only_string(example["solution"]))
    str1, str2 = pred, answer
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss2 = _strip_string(str2)
        # if verbose:
            # print(ss1, ss2)
        return str1 == ss2
    except:
        return str1 == str2


dataset2scorer = {
    "gsm": scorer_number,
    "coin_flip": scorer_string,
    "date_understanding": scoere_date_understanding,
    # "tabmwp": scorer_tabmwp,
    "tabmwp": scorer_tabwmp_official,
    "asdiv": scorer_number,
    "commonsenseqa": scorer_commonsenseqa,
    "strategyqa": scorer_string,
    "aqua": scorer_aqua,
    "MATH": scorer_MATH,
    "last_letters": scorer_string,
}
