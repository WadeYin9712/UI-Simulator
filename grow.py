"""
Code for UI-Simulator-Grow: customizing the trajectory data based on the evaluation results.
"""
from actions.action import *
from simulator import *
import json
import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from converter import WebArenaConverter
import numpy as np
from utils import call_llm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset

def compute_loss_raw(args,
                model, 
                tokenizer, 
                cur_state, 
                instruction, 
                thought,
                action, 
                task, 
                step_history,
                max_length=4096):
    

    # compute the loss of the step
    if args.domain == 'web':
        prompt_template = '''Current webpage:
{}

Goal: {}
Browsing history:
{}'''
        # call the model to get the loss
        with open('system_prompts/inference/webarena.txt', 'r') as f:
            sys_prompt = f.read()
    elif args.domain == 'android':
        prompt_template = '''Current page:
{}

Goal: {}

Browsing history:
{}'''
        with open('system_prompts/inference/android.txt', 'r') as f:
            sys_prompt = f.read()
    prompt = prompt_template.format(cur_state, instruction, step_history)

    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Set to True to get token IDs directly
        add_generation_prompt=True  # Appends assistant's prompt tag
    )

    
    output_template = '''Thought: {}
Action: {}
Step Summary: {}'''
    target = output_template.format(thought, str(action), task)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    target_ids = tokenizer(target, return_tensors="pt").to(model.device)

    full_input = formatted_prompt+ target
    tokenized = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)

    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    labels = input_ids.clone()

    prompt_length = len(tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_length)['input_ids'][0])
    labels[:, :prompt_length] = -100  # Mask the prompt part

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss.cpu()):
            print("NaN loss found!")

    return loss.item()


def compute_loss_traj(args, model, tokenizer, traj):
    '''
    Compute the loss of the trajectory.
    Return:
    loss: the average loss of the trajectory
    '''
    traj_path = os.path.join(traj, 'trajectory.pkl')
    with open(traj_path, 'rb') as f:
        traj_data = pickle.load(f)

    if os.path.exists(os.path.join(traj, 'instruction.txt')):
        with open(os.path.join(traj, 'instruction.txt'), 'r') as f:
            instruction = f.read().strip()

    losses = []
    for step in traj_data:
        if len(step) == 5:
            cur_state, thought, action, task, step_history = step
        else:
            cur_state, thought, action, task, step_history, instruction = step
        if cur_state:
            # compute the loss of the step
            loss = compute_loss_raw(args, model, tokenizer, cur_state, instruction, thought, action, task, step_history, max_length=tokenizer.model_max_length)
            losses.append(loss)

    return np.mean(losses) if losses else 0.0

def evaluate_on_dev_raw_set(args, model, tokenizer, root):
    '''
    Return:
    results: a list of dictionaries, each dictionary contains the following keys:
    - traj_path: the id of the trajectory
    - loss: the score of the step
    '''
    model.eval()
    results = []
    trajs = [p for p in os.listdir(root) if not p.startswith('.')]
    trajs = [os.path.join(root, traj) for traj in trajs]
    for i, traj in enumerate(trajs):
        # compute traj loss
        loss = compute_loss_traj(args, model, tokenizer, traj)
        results.append({
            'traj_path': traj,
            'loss': loss,
        })
    return results


def state_insight(cur_state, instruction, action):
    # call teacher model to extract insight for generating the step in next iteration
    with open('system_prompts/data_collection/state_insight.txt', 'r') as f:
        system_prompt = f.read()
    prompt_template = '''Current webpage:
{}

Thought:
'''
    prompt = prompt_template.format(cur_state, instruction, action)
    # call the teacher model to get the insight
    response = call_llm(prompt, system_prompt)
    insight = response
    if 'scroll' in action:
        insight = insight + ' The current state is incomplete for proceeding to the next step of the task.'
    return insight

def synthesize_new_web_summary(insight, instruction, step_history):
    # synthesize the data based on the insight
    with open('system_prompts/data_collection/customize_state.txt', 'r') as f:
        system_prompt = f.read()
    prompt_template = '''Input:
Analysis of the current webpage:
{}

Reference Instruction: {}
Reference browsing history: {}

Output:'''
    prompt = prompt_template.format(insight, instruction, '\n'.join(step_history))
    
    # call the teacher model to get the customized data
    response = call_llm(prompt, system_prompt)

    return response


def thought_action_gen_with_ref(args, state, guide, step_history, ref_thought, ref_action, ref_task):
    if args.domain == 'web':
        with open('element_select_example/state_example_webarena.json', 'r') as f:
            example_state = json.load(f)

        example_guide = 'Create a new project or issue.'
        converter = WebArenaConverter()
        example_state = converter.convert_venv_to_real_env(example_state)
        if isinstance(state, list):
            state = converter.convert_venv_to_real_env(state)
        with open('element_select_example/step_history_example.txt', 'r') as f:
            example_step_history = f.read()

        with open(f'system_prompts/{args.domain}_data_collection/act_with_ref.txt', 'r') as f:
            sys_prompt = f.read()

        sys_prompt = sys_prompt.format(example_guide, example_state, example_step_history)
    else:
        with open(f'system_prompts/{args.domain}_data_collection/act_with_ref.txt', 'r') as f:
            sys_prompt = f.read()
        indexed_state = ''
        for i, line in enumerate(state.strip().split('\n')):
            indexed_state += f"Element {i}: {line}\n"
    prompt = '''
Input:
Guide: {}
Current state: {}
Previous steps: {}

Reference thought: {}
Reference action: {}
Reference task: {}
\n'''.format(guide, indexed_state, '\n'.join(step_history), ref_thought, ref_action, ref_task)

    response = call_llm(prompt, sys_prompt, temperature=0.3)
    print(response + "\n")
    
    action, thought, task = "", "", ""
    thought = response[response.find("Thought: "):response.find("Action: ")].split("Thought: ")[1].strip()
    action = response[response.find("Action: "):response.find("Task: ")].split("Action: ")[1].strip()
    task = response[response.find("Task: "):].split("Task: ")[1].strip()

    if 'stop' in action:
        task = 'Stop'
    
    return thought, action, task

def propose_customized_task(args, state, ref_instruction, ref_browsing_history):
    # call the teacher model to get the new task
    with open(f'system_prompts/{args.domain}_data_collection/propose_customized_task.txt', 'r') as f:
        system_prompt = f.read()
    if not ref_browsing_history:
        ref_browsing_history = ['None']
    prompt_template = '''Current webpage:
{}

Reference task:
{}

Reference browsing history:
{}

Task:'''
    prompt = prompt_template.format(state, ref_instruction, '\n'.join(ref_browsing_history))

    # call the teacher model to get the new task
    response = call_llm(prompt, system_prompt)

    # post-process the response
    new_instruction = response[:response.find('New browsing history:')].strip()
    new_browsing_history = response[response.find('New browsing history:') + len('New browsing history:'):].strip()

    new_browsing_history = new_browsing_history.split('\n') if 'None' not in new_browsing_history else []
    # keep the characters after the first '. '
    new_browsing_history = [h.split('. ')[1].strip() if '. ' in h else h.strip() for h in new_browsing_history]


    return new_instruction, new_browsing_history

def customize_rag_data(args, cur_state, instruction, thought, action, task, step_history):
    # generate a new state based on the current state
    if args.domain == 'web':
        simulator = Simulator(webarena_mode=True)
    elif args.domain == 'android':
        simulator = AndroidSimulator()
    new_state = simulator.perturb_state(cur_state).strip()

    return_state = new_state
    if args.domain == 'android':
        new_state = simulator.get_valuable_short_state(new_state)

    # generate new task goal based on the new state and the browsing history
    new_instruction, new_step_history = propose_customized_task(args, new_state, instruction, '\n'.join(step_history))

    # generate more diverse task instruction
    # generate the new action based on the new state
    # but keeping the original action type
    thought, action, task = thought_action_gen_with_ref(args, new_state, new_instruction, new_step_history, thought, action, task)

    return return_state, new_instruction, thought, action, task, new_step_history

def process_traj(i, traj, args):
    path = traj['traj_path']

    try:
        with open(os.path.join(path, 'trajectory.pkl'), 'rb') as f:
            traj_data = pickle.load(f)
        if os.path.exists(os.path.join(path, 'instruction.txt')):
            with open(os.path.join(path, 'instruction.txt'), 'r') as f:
                instruction = f.read().strip()
    except Exception as e:
        print(f"Error reading files for trajectory {i}: {e}")
        return
    for j in range(args.cnt_per_traj):
        new_traj = []
        for step in traj_data:
            if len(step) == 5:
                cur_state, thought, action, task, step_history = step
            else:
                cur_state, thought, action, task, step_history, instruction = step
            if cur_state:
                # try:
                new_state, new_instruction, new_thought, new_action, new_task, new_step_history = customize_rag_data(
                    args, cur_state, instruction, thought, action, task, step_history
                )
                new_traj.append((new_state, new_thought, new_action, new_task, new_step_history, new_instruction))
                # except Exception as e:
                #     print(f"Error in customization at traj {i}, sample {j}: {e}")
                #     continue
                # breakpoint()

        # Use i * repeat + j for unique naming
        cnt_id = i * args.cnt_per_traj + j
        directory = f'customization/rewrite_android/{args.model_name}_rag_based/traj_{cnt_id}'
        os.makedirs(directory, exist_ok=True)

        with open(f'{directory}/trajectory.pkl', 'wb') as f:
            pickle.dump(new_traj, f)

def main(args):

    # evaluate the agent on the dev set
    model_id = args.model_id # placeholder

    base_name = os.path.basename(model_id)

    if args.run_eval:
        # load the model
        
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if args.domain == 'android':
            tokenizer.model_max_length = 12288

        # evaluation
        eval_results = evaluate_on_dev_raw_set(args, model, tokenizer, args.root)
        # print("*" * 20)
        # print(eval_results)
        # print("*" * 20)
        if args.save_eval_results:
            with open(os.path.join(f'./ui_grow/{args.domain}', f'{base_name}_eval_results.json'), 'w') as f:
                json.dump(eval_results, f)
    else:
        with open(os.path.join(f'./ui_grow/{args.domain}', f'{base_name}_eval_results.json'), 'r') as f:

            eval_results = json.load(f)
    
    # get 25 - 75% of the eval results
    eval_results = sorted(eval_results, key=lambda x: x['loss'], reverse=True)

    # Prepare list of valid trajectories
    valid_trajs = [(i, traj) for i, traj in enumerate(eval_results)
                if len(eval_results) * 0.25 <= i <= len(eval_results) * 0.75]
    
    # ThreadPool only over the outer loop
    with ThreadPoolExecutor(max_workers=min(16, len(valid_trajs))) as executor:
        list(tqdm(executor.map(lambda p: process_traj(*p, args), valid_trajs), total=len(valid_trajs)))

    return eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='the root path of the dev set for evaluation', required=True)
    parser.add_argument('--model_id', type=str, help='the model id for evaluation', required=True)
    parser.add_argument('--level', type=str, help='grain level for traj customization', choices=['step', 'traj'], default='traj')
    parser.add_argument('--cnt_per_traj', type=int, default=4, help='number of steps to customize per traj')
    parser.add_argument('--save_eval_results', action='store_true', help='whether to save the evaluation results')
    parser.add_argument('--rag_based', action='store_true', help='whether to use rag-based customization')
    parser.add_argument('--run_eval', action='store_true', help='whether to rerun the evaluation')
    parser.add_argument('--model_name', type=str, help='model name for customization')
    parser.add_argument('--domain', type=str, choices=['android', 'web'], default='web', help='domain for customization')
    parser.add_argument('--task_perturb', action='store_true', help='whether to perturb the task instruction')

    args = parser.parse_args()
    results = main(args)