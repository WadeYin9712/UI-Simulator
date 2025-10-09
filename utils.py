import openai
import os
from openai import OpenAI
from Levenshtein import distance
from rank_bm25 import BM25Okapi
import json
import inspect
import time
import re
import actions

def match_action(action: str):
    # Match the class name and all [arg] blocks
    match = re.match(r'(\w+)((?:\s*\[[^\[\]]+\])*)$', action)
    if not match:
        raise ValueError("String format is incorrect")
    
    class_name = match.group(1).lower()
    args_block = match.group(2)

    # Extract arguments from [arg1] [arg2] format
    args = re.findall(r'\[([^\[\]]+)\]', args_block)

    # Convert known types (you can expand this for other cases)
    if class_name == 'input_text':
        args[0] = str(args[0])
        args[1] = str(args[1])
    elif class_name == 'click' or class_name == 'open_app':
        args[0] = str(args[0])
    elif class_name == 'scroll':
        args[0] = str(args[0])
    # No args needed for navigate_back, navigate_home, wait, keyborad_enter

    cls = actions.ACTIONS_NAME.get(class_name)
    if not cls:
        raise ValueError(f"Class {class_name} not found")

    action_instance = cls(*args)
    print(action_instance)
    return action_instance

def match_webarena_action(action: str):
    # Validate the input string format
    match = re.match(r'(\w+)\s*(\[(.*)\])*', action)
    if not match:
        raise ValueError("String format is incorrect")
    
    class_name = match.group(1).lower()
    if class_name != 'type':
        args_str = match.group(2)
        # print("class:", class_name)
        
        # Convert string to a list of arguments
        args = eval(f"{args_str}")  if class_name not in ['goto', 'scroll', 'press'] else [args_str[1:-1]]
    else:
        groups = match.groups()
        # print(match)
        # print(groups)
        args = [a.replace('[', '').replace(']', '') for a in groups[1].split('] [')]
        # print(args)
        args[0] = int(args[0])
        args[2] = int(args[2])

    if not args:
        args = []
    
    # Dynamically get the class from globals() dictionary
    cls = actions.WEBARENA_ACTIONS_NAME.get(class_name)
    if not cls:
        raise ValueError(f"Class {class_name} not found")
    action = cls(*args)
    return action

def split_state(old_state: str, new_state: str):
    # split the new state into the old elements and newly added elements
    old_state = clean_state(old_state).split('\n')
    new_state = clean_state(new_state).split('\n')
    old_elements = []
    new_elements = []
    for s in new_state:
        if s not in old_state:
            new_elements.append(s)
        elif s:
            old_elements.append(s)
    return old_elements, new_elements

def shrink_state(old_state: str, keep_num=100):
    # cut the old state to have less than 100 elements
    tmp = old_state.split('\n')
    if len(tmp) > keep_num:
        tmp = tmp[:keep_num]
    old_state = '\n'.join(tmp)
    return old_state

def filter_transitions(domain: str, batch_size: int = 10):
    root = f'intermediate_states/{domain}'
    files = [f for f in os.listdir(root) if f.endswith('.json')]
    
    saved_action_histories = []
    for file in files:
        with open(f'{root}/{file}', 'r') as f:
            data = json.load(f)
    
        # filter out consecutive duplicate actions
        action_history = data["action_history_list"]

        last_action = None
        new_action_history = []
        for action in action_history:
            if action != last_action:
                new_action_history.append(action)
                last_action = action
        data["action_history_list"] = new_action_history
        # filter out duplicate transitions
        if new_action_history not in saved_action_histories:
            saved_action_histories.append(new_action_history)
            # Save updated files
            with open(f'{root}/{file}', 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(f"Duplicate transition found in {file}")
            # move the file to a different folder
            os.makedirs(f'{root}_dep', exist_ok=True)
            os.rename(f'{root}/{file}', f'{root}_dep/{file}')

def find_element_cleaned(state: str, element_id):
    if '\n\n' in state:
        state = state.split('\n\n')[1].split('\n')
    else:
        state = state.split('\n')
    for s in state:
        if f'[{element_id}]' in s:
            # return the cleaned version of s
            line = ' '.join(s.split()[1:])
            end_idx = line.rfind("'") + 1
            if end_idx < len(line):
                return line[:end_idx]
            return line.strip()
    return None

def clean_state(state: str):
    
    if '\n\n' in state:
        clean_state = state.split('\n\n')[1].split('\n')
    else:
        clean_state = state.split('\n')

    clean_state = [' '.join(s.split()[1:]) for s in clean_state]

    for i, s in enumerate(clean_state):
        end_idx = s.rfind("'") + 1
        if end_idx < len(s):
            clean_state[i] = s[:end_idx]
    clean_state = '\n'.join(clean_state)
    return clean_state

def BM25_ranking(query, corpus, top_n = 5):
    # find if there are exact matches
    top_indices = []
    for i, doc in enumerate(corpus):
        if query == doc:
            top_indices.append(i)
    if top_indices:
        return False, top_indices
    
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-top_n:][::-1]
    return True, top_indices


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} use {end_time-start_time:.3f}s")
        return result
    return wrapper


def call_llm(prompt, sys_prompt, model='gpt-4o-mini', stop=None, return_json=False, max_tokens=None, temperature=0.5):
    client = OpenAI(
        organization=os.environ['OPENAI_ORG_ID'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" } if return_json else openai.NOT_GIVEN,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature
    )

    # get caller's function name
    caller = inspect.stack()[1].function
    usage = completion.usage
    print(f"Function ##{caller}## called with usage: Input token:{usage.prompt_tokens}, Output token:{usage.completion_tokens}")
    return completion.choices[0].message.content
