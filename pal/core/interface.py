# Copyright 2022 PAL Authors. All rights reserved.
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

import io
import signal
from contextlib import redirect_stdout
from typing import Any, Callable, List, Optional
from collections import Counter

from .runtime import GenericRuntime
from .backend import call_gpt, call_chat_gpt


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TextInterface:
    
    def __init__(
        self,
        model: str = 'code-davinci-002',
        answer_prefix: str = 'The answer is:',
        stop: str = '\n\n\n',
        extract_answer: Optional[Callable[[str], Any]] = None,
        verbose: bool = False,
    ):
        self.history = []
        self.answer_prefix = answer_prefix
        self.extract_answer_fn = extract_answer
        self.stop = stop
        self.model = model
        self.verbose = verbose
        
    def clear_history(self):
        self.history = []
    
    def extract_answer(self, gen: str):
        if self.extract_answer_fn:
            return self.extract_answer_fn(gen)
        last_line = gen.strip().split('\n')[-1]
        return last_line[len(self.answer_prefix):].strip()
    
    def run(self, prompt, temperature=0.0, top_p=1.0, majority_at=None, max_tokens=512):
        gen = call_gpt(prompt, model=self.model, stop=self.stop, 
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at)
        self.history.append(gen)
        return self.extract_answer(gen)
        

class TextChatInterface(TextInterface):
    
    def __init__(self, *args, system_message="You are a helpful assistant.", **kwargs):
        super().__init__(*args, **kwargs)
        self.system_message = system_message

    def clear_history(self):
        self.history = []
    
    def extract_answer(self, gen: str, example=None):
        if self.extract_answer_fn:
            return self.extract_answer_fn(gen, example=example)
        last_line = gen.strip().split('\n')[-1]
        return last_line[len(self.answer_prefix):].strip()
    
    def run(self, prompt=None, temperature=0.0, top_p=1.0, majority_at=None, max_tokens=512, example=None, messages=None, api_source="openai"):
        gens = call_gpt(messages=messages, prompt=prompt, model=self.model, stop=self.stop, 
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, 
            system_message=self.system_message, api_source=api_source)
        # gen = gens[0]
        self.history.append(gens)
        if self.verbose:
            for gen in gens:
                print(gen)
        results = []
        for gen in gens:
            results.append(self.extract_answer(gen, example=example))
        counter = Counter(results)
        return counter.most_common(1)[0][0]


SYSTEM_MESSAGES = 'You are a helpful python programmer.'
class ProgramInterface:
    
    def __init__(
        self,
        model: str = 'code-davinci-002',
        runtime: Optional[Any] = None,
        stop: str = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        verbose: bool = False,
        system_message: str = SYSTEM_MESSAGES,
        extract_answer: Optional[Callable[[str], Any]] = None,
    ) -> None:

        self.model = model
        self.runtime = runtime if runtime else GenericRuntime()
        self.history = []
        self.stop = stop
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.verbose = verbose
        self.system_message = system_message
        self.extract_answer = extract_answer
        
    def clear_history(self):
        self.history = []
    
    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]
    
    def generate(self, prompt: str, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int =None, ):
        gens = call_gpt(prompt, model=self.model, stop=self.stop, 
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, system_message=self.system_message)
        if self.verbose:
            for gen in gens:
                print(gen)
        code = self.process_generation_to_code(gens)
        self.history.append(gens)
        return code
    
    def execute(self, code: Optional[List[str]] = None):
        code = code if code else self.code
        if self.get_answer_from_stdout:
            program_io = io.StringIO()
            with redirect_stdout(program_io):
                self.runtime.exec_code('\n'.join(code))
            program_io.seek(0)
            return program_io.readlines()[-1]
        elif self.answer_symbol:
            self.runtime.exec_code('\n'.join(code))
            return self.runtime._global_vars[self.answer_symbol]
        elif self.answer_expr:
            self.runtime.exec_code('\n'.join(code))
            return self.runtime.eval_code(self.answer_expr)
        else:
            self.runtime.exec_code('\n'.join(code[:-1]))
            return self.runtime.eval_code(code[-1])
    
    def run(self, prompt: str, time_out: float =10, temperature: float =0.0, top_p: float =1.0, 
            max_tokens: int =512, majority_at: int =None, example: Optional[dict] = None, api_source="openai"):
        code_snippets = self.generate(prompt, majority_at=majority_at, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        
        results = []
        for idx, code in enumerate(code_snippets):
            # with timeout(time_out):   # Report error when using multithreading 
            try:
                exec_result = self.execute(code)
                exec_result = str(exec_result)    # In case the result is not a number, e.g. a tuple or a list 
                # Append the result to the code snippet
                self.history[-1][idx] = self.history[-1][idx] + "\n\n" + f"print(solution()) \n # Execution result: {str(exec_result)} \n Therefore, the final answer is \\boxed" + "{" + str(exec_result) + "}."
            except Exception as e:
                print(e)
                self.history[-1][idx] = self.history[-1][idx] + "\n\n" + "ERROR OCCURRED WHEN EXECUTING THE CODE."
                continue
            if self.extract_answer:
                exec_result = self.extract_answer(exec_result, example=example)
            results.append(exec_result)
        
        if len(results) == 0:
            print('No results was produced. A common reason is that the generated code snippet is not valid or did not return any results.')
            return None
        
        counter = Counter(results)
        return counter.most_common(1)[0][0]
    
    
SYSTEM_MESSAGES = 'You are a helpful python programmer.'
class ProgramChatInterface(ProgramInterface):
    def __init__(self, *args, system_message: str = SYSTEM_MESSAGES, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_message = system_message
        
    def generate(self, prompt: str, temperature: float = 0, top_p: float = 1, max_tokens: int = 512):
        messages =[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': prompt}]
        gens = call_chat_gpt(messages, model=self.model, stop=self.stop, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        if self.verbose:
            for gen in gens:
                print(gen)
        self.history.append(gens)
        return self.process_generation_to_code(gens)
        
    def process_generation_to_code(self, gens: str):
        if '```python' in gens:
            gens = gens.split('```python')[1].split('```')[0]
        elif '```' in gens:
            gens = gens.split('```')[1].split('```')[0]
            
        return gens.split('\n')
    
    def run(self, prompt: str, time_out: float = 10, temperature: float = 0, top_p: float = 1, max_tokens: int = 512):
        code = self.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        # with timeout(time_out):
        exec_result = None
        try:
            exec_result = self.execute(code)
        except Exception as e:
            print(e)
        return exec_result
