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


ACTS = ["vanilla", "cot", "ps", "pal"]

######################## SYSTEM MESSAGES ########################
VANILLA_SYS = "You are a helpful assistant."
COT_SYS = "You are a helpful assistant. Lets's think step by step to answer the request."
PS_SYS = "You are a helpful assistant. Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan, solve the problem step by step, and give the ultimate answer. Please explicitly generate the mentioned process."
PAL_SYS = "You are a helpful assistant. Lets's continue coding to answer the request"


######################## Methodology ########################
act2methodology = {
    "vanilla": "",
    "cot": "Let's think step by step.",
    "ps": "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan, solve the problem step by step, and give the ultimate answer. Please explicitly generate the mentioned process: [Problem Understanding], [Plan], [Solving/Calculations], [Answer]. in your response.",
}

TEXT_PROMPT = """
Q: {question}

Your answer should be in the form \\boxed{{answer}}.
"""

TEXT_PROMPT_CHOIECES = """
Q: {question}

Your answer should be in the form \\boxed{{choice}}, e.g. \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}. There is only one correct choice.
"""

TEXT_PROMPT_YES_NO = """
Q: {question}

Your answer should be either \\boxed{{yes}} or \\boxed{{no}}, in the form \\boxed{{answer}}.
"""

TEXT_PROMPT_GSM = """
Q: {question}

Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.
"""

TEXT_PROMPT_MATH = """
Q: {question}

Your final answer should be a number or choice, in the form \\boxed{{answer}}, at the end of your response.
"""

TEXT_PROMPT_COIN = """
Q: {question}

Your answer should be \\boxed{{yes}} or \\boxed{{no}}, in the form \\boxed{{answer}}.
"""

PAL_PROMPT_GSM = '''
#!/bin/python3

import math
import numpy as np
import statistics
import sympy as sp

#################### Task Instruction ####################
# You will write python program to solve math problems. 
# You will only write code blocks. 
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   """<question>"""
#   <your code>
#   result = <your result>
#   return result
##########################################################

# To better solve the probelm, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) You can use the imported python package to solve the problem.

# Q: {question}

# Your defined "solution()" function with comments here. 
'''.strip()


PAL_PROMPT_COIN = '''
# You will write python program to solve the coin flip problems. 
# You will only write code blocks. 
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   """
#   :return: "yes" or "no" 
#   """
#   <your code>
#   result = <your result>
#   assert result in ["yes", "no"]
#   return result # "yes" or "no"

# To better solve the probelm, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) If you use the library, you must import it by `import {{module}}` in the first line of your code block.


#!/bin/python3

# Q: {question}
# Your "solution()" function here.  Remember only to return "yes" or "no".
'''.strip()


PAL_PROMPT_DATE = '''
#!/bin/python3

import datetime
import dateutil.relativedelta as relativedelta

#################### Task Instruction ####################
# You will write python program to solve the date understanding problems. 
# You will only write code blocks. 
# Please generate your code block in `def solution()` function, and thus it can be executed by python interpreter. You don't need to call `solution()` function because it will be called by the system.
# The concrete format of `solution()` is as follows:
# def solution():
#   <your code>
#   return result # "MM/DD/YYYY" by strftime('%m/%d/%Y').
##########################################################

# [TIPS]: To better solve the problem, 1) you can define variables first and then use them in your code block. 2) You can add comments to the code. 3) You can use relevant libraries, including `datetime.datetime`, `dateutil.relativedelta.relativedelta`, among others. 4) NOTICE! The 'today' in the [QUESTION] does not mean the date of the execution of the code, so you don't need and can NOT use the `datetime.datetime.now()` or `datetime.date.today()` to get the today date.

# [QUESTION]: "{question}"
# [RESPONSE FORMAT] You need to infer the date from context, and return the result in the format of `MM/DD/YYYY`.
# Your defined "solution()" function here. Remember you DON'T need and CAN NOT use the `datetime.datetime.now()` or `datetime.date.today()`.
'''.strip()


TEXT_PROMPT_DATE = """
Can you solve the following date understanding problem?
[TASK INSTRUCTION] You need to infer the date from context, and return the result in the format of `MM/DD/YYYY`.
[QUESTION]: {question}
Your final answer should be a date, in the format of \\boxed{{MM/DD/YYYY}}, e.g. \\boxed{{05/01/2022}}, at the end of your response.
"""


TEXT_PROMPT_DATE = """
Q: {question}

Your answer should be a date, in the format of \\boxed{{MM/DD/YYYY}}, e.g. \\boxed{{05/01/2022}}.
"""


dataset2prompt = {
    "gsm": {
        "pal": PAL_PROMPT_GSM,
        "text": TEXT_PROMPT_GSM,
    },
    "coin_flip": {
        "pal": PAL_PROMPT_COIN,
        "text": TEXT_PROMPT_COIN,
    },
    "date_understanding": {
        "pal": PAL_PROMPT_DATE,
        "text": TEXT_PROMPT_DATE,
    },
    "asdiv": {
        "pal": PAL_PROMPT_GSM,
        "text": TEXT_PROMPT_GSM,
    },
    "commonsenseqa": {
        "pal": None,
        "text": TEXT_PROMPT_CHOIECES,
    },
    "strategyqa": {
        "pal": None,
        "text": TEXT_PROMPT_YES_NO,
    },
    "aqua": {
        "pal": None,
        "text": TEXT_PROMPT_CHOIECES,
    },
    "MATH":{
        "pal": None,
        "text": TEXT_PROMPT_MATH,
    },
    "last_letters": {
        "pal": None,
        "text": TEXT_PROMPT,
    },
}


class Prompter:
    """ The concise/default version of PrompterV2. Only contains the vanilla and cot. """
    def __init__(self, dataset="gsm") -> None:
        self.dataset = dataset

    def get_sys_message(self, sys_type):
        return "You are a helpful assistant."
    
    def _build_commonsenseqa_question(self, example):
        json_res = example
        choice = "Answer Choices:"
        for c in json_res["question"]["choices"]:
            choice += " ("
            choice += c["label"]
            choice += ") "
            choice += c["text"]
        q = json_res["question"]["stem"].strip() + " " + choice
        return q

    def _build_aqua_question(self, example):
        json_res = example
        choice = "(" + "(".join(json_res["options"])
        choice = choice.replace("(", " (").replace(")", ") ")
        choice = "Answer Choices:" + choice
        q = json_res["question"].strip() + ' ' + choice
        return q

    def build_prompt(self, example, sys_type='vanilla', read_times=1):
        if self.dataset in ["commonsenseqa"]:
            question = self._build_commonsenseqa_question(example)
        elif self.dataset in ["aqua"]:
            question = self._build_aqua_question(example)
        elif self.dataset in ["MATH"]:
            question = example["problem"]
        else:
            question = example["input"]
        reread_prompt = "Read the question again: " 
        if sys_type == "pal":
            reread_prompt = "# " + reread_prompt
        input = f" \n\n{reread_prompt}".join([question] * read_times)
        prompt_type = "pal" if sys_type == "pal" else "text"
        prompt = dataset2prompt[self.dataset][prompt_type]
        prompt = prompt.format(question=input)
        # Add methodology
        if sys_type != "pal":
            prompt += f"\nA: {act2methodology[sys_type]}"
        return prompt


dataset2prompter = {
    "gsm": Prompter,
    "coin_flip": Prompter,
    "date_understanding": Prompter,
    "asdiv": Prompter,
    "commonsenseqa": Prompter,
    "strategyqa": Prompter,
    "aqua": Prompter,
    "MATH": Prompter,
    "last_letters": Prompter,
}





