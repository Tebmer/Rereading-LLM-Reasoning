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


import copy
import datetime
from typing import Any, Dict
import dateutil.relativedelta
import math
import sympy
from sympy import symbols, Eq, solve
import numpy as np
import statistics
import pandas as pd


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        
        for c in self.HEADERS:
            self.exec_code(c)
        
    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

    
class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        # 'relativedelta': dateutil.relativedelta.relativedelta
        'relativedelta': dateutil.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

    
class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class MathRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "math": math,
        "sympy": sympy,
        "symbols": symbols,
        "Eq": Eq,
        "solve": solve,
        "sp": sympy,
        "numpy": np,
        "np": np,
        "statistics": statistics,
    }


class TablemwpRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "np": np,
        "numpy": np,
        "statistics": statistics,
        "math": math,
        "pandas": pd,
        "pd": pd,
    }


dataset2runtime = {
    'date_understanding': DateRuntime(),
    'color': ColorObjectRuntime(),
    'gsm': MathRuntime(),
    'tabmwp': TablemwpRuntime(),
    'coin_flip': GenericRuntime(),
    'asdiv': MathRuntime(),
}