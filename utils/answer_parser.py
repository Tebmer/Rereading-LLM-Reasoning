import re 
import random
import numpy as np
random.seed(123)
from utils.util import last_boxed_only_string, remove_boxed, _strip_string


# 1. TODO: Normalize the extracted answer before output.

def extract_the_fist_number(input_str, example=None):
    """ Extract the first number in the input string. """
    # Ref. https://github.com/composable-models/llm_multiagent_debate/blob/9b32e0971105a3ada91706d130582bde9b9a855c/gsm/eval_gsm.py#L46C1-L53C16
    pattern = r"[\-+]?\d*[\.,/]?\d+"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[0]

    return None

def extract_the_last_number(input_str, example=None):
    """ Extract the last number in the input string. """
    # Ref. https://github.com/composable-models/llm_multiagent_debate/blob/9b32e0971105a3ada91706d130582bde9b9a855c/gsm/eval_gsm.py#L46C1-L53C16
    pattern = r"[\-+]?\d*[\.,/]?\d+"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def parse_answer_number(input_str, example=None):
    # https://github.com/composable-models/llm_multiagent_debate/blob/9b32e0971105a3ada91706d130582bde9b9a855c/gsm/eval_gsm.py#L55C1-L66C20
    
    candidate_answer = None
    # 1. Extract candidate answer
    # 1.1 Extract {number}
    # pattern = r"\{([0-9.,$]*)\}"
    pattern = r"boxed\{(.*?)\}"  # non-greedy
    matches = re.findall(pattern, input_str)
    if matches:
        candidate_answer = matches[-1].strip()

    # 1.2 Extract "answer is "
    if candidate_answer is None:
        text = input_str.split('answer is ')
        if len(text) > 1:
            candidate_answer = text[-1].strip()
    
    # 1.3 If no answer in "{}", we try to extract the last sentence of the last paragraph of generation as the candidate answer.
    if candidate_answer is None:
        last_paragraph = input_str.split("\n")[-1]
        candidate_answer = last_paragraph.split(". ")[-1]

    # 2. Extract the answer from candidate answer
    if candidate_answer is not None:
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', candidate_answer)  # The first number 
        if match:
            return match.group().replace(',', '')
        # match = re.findall(r'[\-+]?\d*[\.,/]?\d+', candidate_answer)  # The last number 
        # if match:
        #     return match[-1].replace(",", "").strip()
        
    # 3. If still no answer, we directly extract the last number in the whole generation as the answer.
    solution = extract_the_last_number(input_str, example=None)

    try:
        solution = float(solution)
    except:
        solution = None
        
    return solution


def parse_answer_yes_or_no(input_str, example=None):
    """ Parse the answer of yes/no question packed by \\boxed{}. """
    pattern = r"\{([a-zA-Z]*)\}"
    matches = re.findall(pattern, input_str)
    for match_str in matches[::-1]:
        if "yes" in match_str.lower():
            return "yes"
        elif "no" in match_str.lower():
            return "no"

    # If no answer in "{}", we try to extract the last occurence of "yes" or "no" in the whole input.
    input_str = input_str.lower()
    pattern = r"(yes|no)"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None
    

def parse_answer_date(input_str, example=None):
    """ Parse the answer of date packed by \\boxed{} and in the format of 'MM/DD/YYYY'. """
    def normarlize_date(answer):
        splits = answer.split("/")
        if len(splits) != 3:
            return answer
        m, d, y = splits
        if len(d) == 1:
            d = "0" + d
        if len(m) == 1:
            m = "0" + m
        if len(y) == 2:
            y = "20" + y
        return "/".join([m, d, y])
    # pattern of three times of 'MM/'
    pattern = r"\{([0-9/]*)\}"
    matches = re.findall(pattern, input_str)
    for match_str in matches[::-1]:
        answer = normarlize_date(match_str.lower())
        return answer

    # If no answer in "{}", we try to extract the last occurence of MM/DD/YYYY in the whole input.
    input_str = input_str.lower()
    pattern = r"(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/([0-9]{4})"
    matches = re.findall(pattern, input_str)
    if matches:
        # Combine the match strings in the tuple 
        answer = "/".join(matches[-1])
        return normarlize_date(answer)
    return None


def parser_string_answer(input_str, example=None):
    """ Parse the answer of date packed by  '<|startofanswer|> answer <|endofanswer|>', the answer could be number or string"""
    pattern = r"<\|startofanswer\|>(.*)<\|endofanswer\|>"
    matches = re.findall(pattern, input_str)
    for match_str in matches[::-1]:
        match_str = match_str.strip()
        if match_str:
            return match_str
    return None


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0


def extract_prediction(output, options, option_inds):
    # https://github.com/lupantech/PromptPG/blob/9d3246700f7f74c01eacf2aabf59f5b9b944249b/run_gpt3/utilities.py

    # Extract the answer packed by "<|startofanswer|>" and "<|endofanswer|>""
    # pattern = r"<\|startofanswer\|>(.*)<\|endofanswer\|>"
    # Extract content packed by "{}" from \\boxed{answer}
    # pattern = r"\{([^\{\}]*)\}"
    pattern = r"\{(.*)\}"
    matches = re.findall(pattern, output)
    if matches:
        output = matches[-1].strip()

    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)
    output = re.sub(r"(?<![AP]\.M)\.$", "", output)
    output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
    output = re.sub(r"\u2212", "-", output) # the "Minus Sign" (−) in Unicode
    # Remove "\text" in the output and "{" and "}"
    output = re.sub(r"\\text", "", output)
    output = re.sub(r"\{", "", output)
    output = re.sub(r"\}", "", output)

    ## Multi-choice questions
    if options:
        patterns = [r'^\(([A-Za-z])\)$', # "(b)", "(B)"
                    r'^([A-Za-z])$', # "b", "B"
                    r'^([A-Za-z]). ', # "b", "B"
                    r'[Th]he answer is ([A-Z])', # "The answer is B"
                    r'^\(([A-Za-z])\) [\s\S]+$', # "(A) XXXXX"
                    r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$', # "The answer is (B) XXXXX."
                ]
    
        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper() # e.g., "B"   TODO: check if 'res[0]' is correct
                if pred in option_inds:
                    ind = option_inds.index(pred) # 1
                    if ind >= len(options):
                        ind = random.choice(range(len(options)))
                    prediction = options[ind]
                    return prediction

        # find the most similar options
        ## Narrow down the scope of string.
        output = output.split("\n")[-1]
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction
        
    else:
        ## free_text QA problems, numeric answer
        patterns = [
                    r'The answer is ([\d\$\.\,\/\:]+)',  # "The answer is 1,000." Put extracting the number as the first priority.
                    r' ([\d\$\.\,\/\:]+ [AP]\.M\.)', # 7:25 P.M.
                    r'([\-\d\$\.\,\/\:]{0,}[\d]+)', # 14.5
            ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                if prediction.endswith(".") and ".M." not in prediction:
                    prediction = prediction[:-1]
                return prediction

    return output


def normalize_answer(text, unit):
    # https://github.com/lupantech/PromptPG/blob/9d3246700f7f74c01eacf2aabf59f5b9b944249b/run_gpt3/run_gpt3.py#L53
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("\<\|startofanswer\|\>", "", text)
    text = re.sub("\<\|endofanswer\|\>", "", text)
    text = text.strip()
    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)

    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # number, fraction, or decimal
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)

        if result is not None:
            number = int(text)
        elif "/" in text:
            nums = text.split("/")
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r"\.[0]+$", "", number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text


OPTION_INDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
def parse_tabmwp_official(output, example):
    pred = extract_prediction(output, example["choices"], OPTION_INDS)
    pred = normalize_answer(pred, example["unit"])
    return pred



def parse_multiple_choices(input_str, example=None):
    pattern = r"boxed\{(.*?)\}"
    # pattern = r"\{([^\{\}]*)\}"
    # included by "{}" but there is no 
    matches = re.findall(pattern, input_str)
    if matches:
        input_str = matches[-1].strip()
    else:   # Take the last paragraph
        input_str = input_str.split("\n")[-1]
    pred = input_str.strip()

    # Extract (A) or (B) or (C) or (D) or (E)
    pred_answer = re.findall(r'\(([A-E])\)', pred)
    # pred_answer = re.findall(r'A|B|C|D|E', pred)
    # pred_answer = re.findall(r'[A-Z]', pred)
    if pred_answer:
        pred_answer = pred_answer[-1]
        return pred_answer
    
    # Extract A B C D E surrounded by non-alphabetical characters. Should consider the case of the answer in the beginning and the end of the string.
    pred_answer = re.findall(r'(?<![a-zA-Z])[ABCDE](?![a-zA-Z])', pred)
    if pred_answer:
        pred_answer = pred_answer[-1]
        return pred_answer

    # If no choice in answer, we choose the option by similarity 
    label2options = {x["label"]: x["text"] for x in example["question"]["choices"]}
    options = list(label2options.values())
    labels = list(label2options.keys())
    scores = [score_string_similarity(x, input_str) for x in options]
    max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
    pred_answer = labels[max_idx]
    return pred_answer


def parse_multiple_choices_aqua(input_str, example=None):
    pattern = r"boxed\{(.*?)\}"
    matches = re.findall(pattern, input_str)
    if matches:
        input_str = matches[-1].strip()
    else:   # Take the last paragraph
        input_str = input_str.split("\n")[-1]
    pred = input_str.strip()

    # Extract (A) or (B) or (C) or (D) or (E)
    pred_answer = re.findall(r'\(([A-E])\)', pred)
    if pred_answer:
        pred_answer = pred_answer[-1]
        return pred_answer
    
    # Extract A B C D E surrounded by non-alphabetical characters. Should consider the case of the answer in the beginning and the end of the string.
    pred_answer = re.findall(r'(?<![a-zA-Z])[ABCDE](?![a-zA-Z])', pred)
    if pred_answer:
        pred_answer = pred_answer[-1]
        return pred_answer

    # If no choice in answer, we choose the option by similarity 
    options = example["options"]
    labels = OPTION_INDS
    scores = [score_string_similarity(x, input_str) for x in options]
    max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
    pred_answer = labels[max_idx]
    return pred_answer


def parse_MATH_answer(input_str, example=None):
    """ Parse the answer of MATH dataset."""
    # 1. Extract candidate answer
    # Use the brace matching to extract the candidate answer 
    candidate_answer = remove_boxed(last_boxed_only_string(input_str))
    # 2. Extract the answer from candidate answer
    try:
        answer = _strip_string(candidate_answer)
    except:
        answer = candidate_answer
    return answer


def parse_text_answer(input_str, example=None):
    """ Parse the answer of textual answer"""
    # Use regext to extract the answer directly
    pattern = r"boxed\{(.*?)\}"
    matches = re.findall(pattern, input_str)
    if matches:
        input_str = matches[-1].strip()
        return input_str
    else:
        return None


dataset2parser = {
    "gsm": parse_answer_number,
    "coin_flip": parse_answer_yes_or_no,
    "date_understanding": parse_answer_date,
    "tabmwp": parse_tabmwp_official,
    "asdiv": parse_answer_number,
    "commonsenseqa": parse_multiple_choices,
    "strategyqa": parse_answer_yes_or_no,
    "aqua": parse_multiple_choices_aqua,
    "MATH": parse_MATH_answer,
    "last_letters": parse_text_answer,
}


def test_parser():
    # Test the number 
    number_text = ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2", "1,000.", "1,000.0", "1,000.00", "1,000.000", "1,000.0000"]
    for t in number_text:
        print(normalize_answer(t, None))
    
    # Test the string 
    string_text = ["yes", "Hunter", "6:50 P.M.", "6:50 P.M"]
    for t in string_text:
        print(normalize_answer(t, None))



if __name__ == "__main__":
    ############################################# Test parse_answer_number #############################################
    input_str = "To determine the maximum number of books that are both hardcover and fiction, we can use the principle of inclusion-exclusion.\n\nLet:\n- \\( H \\) be the number of hardcover books = 30\n- \\( F \\) be the number of fiction books = 20\n- \\( T \\) be the total number of books = 45\n\nWe can use the formula:\n\\[\n\\text{Maximum of } (H + F) - T\n\\]\n\nCalculating:\n\\[\nH + F = 30 + 20 = 50\n\\]\nSince the total number of books is 45, the maximum number of books that can be both hardcover and fiction is:\n\\[\n50 - 45 = 5\n\\]\n\nHowever, this computation determines the overlap if all books were counted separately. We are really interested in the situation where we find the overlap directly under the given conditions, so instead, we need to check the maximum count without exceeding both individual counts.\n\nTo maximize the number of books that are both hardcover and fiction:\nLet \\( x \\) be the number of books that are both hardcover and fiction. \nFrom the principle of limitations, we know:\n1. The maximum cannot exceed the number of hardcover books: \\( x \\leq 30 \\)\n2. The maximum cannot exceed the number of fiction books: \\( x \\leq 20 \\)\n\nBoth conditions must hold together with the total. To find the limit where both conditions work, consider:\n- Books that are neither hardcover nor fiction cannot be more than \\( T - (H + F - x) \\) and this implies checking against the limits.\n\nWe can add the limits:\n- Maximum intersection occurs when not exceeding any limits, therefore \\( x = \\min(30, 20) = 20 \\).\n\nNext, we check:\nIf we assume \\( x = 20 \\):\n- Hardcover only: \\(30 - 20 = 10\\) (hardcover not fiction)\n- Fiction only: \\(20 - 20 = 0\\) (fully overlapping)\n- Total accounted = \\(20 (both) + 10 (hardcover only) + 0 (fiction only) = 30\\)\n\nThis total indeed fits within the conditions of the bookshelf.\n\nThus the maximum number of books that can be both hardcover and fiction is:\n\\[\n\\boxed{20}\n\\]"
    example = {
        "question":{
            "choices": ["A)10", "B)15", "C)18", "D)20", "E)30"]
        }
    }
    res = parse_multiple_choices(input_str, example)
    print(res)
    
