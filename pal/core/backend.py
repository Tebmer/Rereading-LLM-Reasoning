# Copyright 2022 PAL Authors. All rights reserved.
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

import openai
import time
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

# GPT-3 API
def call_gpt(prompt=None, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None, system_message="You are a helpful assistant.", messages=None, api_source='openai'):  
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
        
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))           
            if model.startswith('gpt-4') or 'turbo' or "claude" in model:
                ans = chat_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            messages=messages,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions,
                            system_message=system_message,
                            api_source=api_source)
            else:
                ans = completions_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions,
                            api_source=api_source)
            completions.extend(ans)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        # except openai.error.RateLimitError as e:
        #     time.sleep(min(i**2, 60))
        # except openai.error.ServiceUnavailableError as e:
        #     time.sleep(min(i**2, 60))
        # except openai.error.APIConnectionError:
        #     time.sleep(min(i**2, 60))
        # except openai.OpenAIError as e: # Including ratelimiterror and serviceunavailableerror
        #  openai.error.RateLimitError and openai.error.ServiceUnavailableError are both subclasses of openai.error.OpenAIError
        except openai.error.InvalidRequestError as e:
            print(e)
            break
        except openai.error.OpenAIError as e:
            time.sleep(min(i**2, 60))
            
    raise RuntimeError('Failed to call GPT API')

def completions_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of, api_source="openai"):
    kwargs = {"model": model} if api_source == 'openai' else {"engine": model}
    ans = openai.Completion.create(
        **kwargs,
        max_tokens=max_tokens,
        stop=stop,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        n=n,
        best_of=best_of)
    return [choice['text'] for choice in ans['choices']]


def chat_api(model, max_tokens, stop, temperature,
            top_p, n, best_of, messages=None, prompt=None, system_message='You are a helpful assistant.', api_source='openai'):
    if messages is None:
        assert prompt is not None
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}]
    kwargs = {"model": model} if api_source == 'openai' else {"engine": model}
    ans = openai.ChatCompletion.create(
        **kwargs,
        max_tokens=max_tokens,
        stop=stop,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n)
    return [choice['message']['content'] for choice in ans['choices']]


def chat_azure_api(model, max_tokens, stop, temperature,
            top_p, n, best_of, messages=None, prompt=None, system_message='You are a helpful assistant.', api_source='openai'):
    """ Call Azure chat API. The argument name of model is 'engine'. """
    if messages is None:
        assert prompt is not None
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}]
    kwargs = {"model": model} if api_source == 'openai' else {"engine": model}
    ans = openai.ChatCompletion.create(
        **kwargs,
        max_tokens=max_tokens,
        stop=stop,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n)
    return [choice['message']['content'] for choice in ans['choices']]


def call_chat_gpt(messages, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=1.0, max_tokens=128):
    wait = 1
    while True:
        try:
            ans = openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                stop=stop,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            return ans.choices[0]['message']['content']
        # except openai.error.RateLimitError as e:
        except openai.OpenAIError as e:
            time.sleep(min(wait, 60))
            wait *= 2
    raise RuntimeError('Failed to call chat gpt')
