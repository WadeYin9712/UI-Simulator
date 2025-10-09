import random
import actions
from actions.action import *
from simulator import *
import copy
import os
import ast
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import re
from utils import *
import traceback
import argparse

############################################
###### Guide related functions ######
############################################

def judge_missed_entity(guide):
    # check if guide is search task or not
    with open('system_prompts/android_data_collection/judge_entity_guide.txt', 'r') as f:
        sys_prompt = f.read()
    prompt = '''
Original task: {}'''.format(guide)
    response = call_llm(prompt, sys_prompt)
    return "Yes" in response or "yes" in response

def specify_entity(guide):
    # print(description)
    # specific entity
    with open('system_prompts/android_data_collection/specific_entity.txt', 'r') as f:
        sys_prompt = f.read()
    prompt = '''
Original task:
{}
Thought: I'll try to'''.format(guide)
    response = call_llm(prompt, sys_prompt, model='gpt-4o-mini')
    guide = re.split(r'(?i)tasks:', response)[1].strip().split('\n')
    guide = [line.split('. ', 1)[1].strip() for line in guide if line and line[0].isdigit()]
    # print(guide)
    guide = random.sample(guide, 1)[0]
    return guide

def replace_entity(guides: list):
    with open('system_prompts/android_data_collection/replace_entity.txt', 'r') as f:
        system_prompt = f.read()
    prompt = '''
Guides:
{}

New guides:\n'''.format('\n'.join(guides))
    response = call_llm(prompt, system_prompt)
    guides = response.split('\n')

    return guides

def judge_guide_completion(guide, prev_steps, cur_state):
    with open('system_prompts/android_data_collection/judge_guide_completion.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
Original task:
{}
Step history:
{}
Current Elements:
{}
    """.format(guide, '\n'.join(prev_steps), cur_state)
    response = call_llm(prompt, system_prompt)
    print(response)
    try:
        answer = response.split("Answer:")[1]
        return "Yes" in answer or "yes" in answer
    except:
        return 'Yes' in response or 'yes' in response

def task_guidance(cur_state: str,
                  prev_guide = None, 
                  prev_step = None, 
                  first_guide=True,
                  num_guides=5):
    # state: a11y tree
    if first_guide:
        # manually provide the candidate tasks / apps

        with open('system_prompts/android_data_collection/app_proposal.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = '''Tasks:\n'''

    else:
        with open('system_prompts/android_data_collection/task_control.txt', 'r') as f:
            sys_prompt = f.read()

        example_guides = [
            "1. Create a new contact'"
            "2. Switch to 'Phone contacts'"
            "3. Open the navigation drawer."
        ]
        example_guides = '\n'.join(example_guides[:num_guides])
        sys_prompt = sys_prompt.format(num_guides, example_guides)

        prompt = '''
Elements:
{}

Original task guide:
{}
Previous steps:
{}

'''.format(cur_state, prev_guide, prev_step)
        prompt += "Thought: Let's think step by step."
    
    response = call_llm(prompt, sys_prompt, model='gpt-4o-mini')
    print("Thought response")
    print(response)
    
    # post-processing
    if first_guide:
        outputs = [r for r in response.strip().split('\n')]
        tasks = [r.split(".", 1)[1].strip() for r in outputs if r[0].isdigit()]
        return tasks
    try:
        thought, task = response.split("Guides:")
    except:
        task = '\n'.join(response.split('\n')[1:])
    task = task.strip()
    if "None" in task:
        return []
    tasks = task.split('\n')
    for i, t in enumerate(tasks):   
        if '. ' in t:
            tasks[i] = t.split('. ')[1]

    return tasks

############################################
###### Reasoning related functions ######
############################################

def propose_reasoning_tasks(cur_state):
    with open('system_prompts/android_data_collection/reasoning_guide.txt', 'r') as f:
        system_prompt = f.read()

    prompt = """
UI:
{}
Thought: Let's think step by step. """.format(cur_state)
    response = call_llm(prompt, system_prompt)
    print("response:", response)
    try:
        thought, questions = response.split("Questions:")
        questions = questions.strip().split('\n')
    except:
        if 'Questions:' in response:
            i1 = response.find("Questions:")
            questions = response[i1 + len("Questions:"):].strip().split('\n')
        else:
            return []

    for i, q in enumerate(questions):
        if '. ' in q:
            if q.split('. ')[1].strip() != '':
                questions[i] = q.split('. ')[1]
    return questions
    

def judge_reasoning(high_level_intent, step_history):
    with open('system_prompts/android_data_collection/judge_reasoning.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
Original task guide: {}
Browsing history:
{}
""".format(high_level_intent, '\n'.join(step_history))
    response = call_llm(prompt, system_prompt)
    print(response)
    return "Yes" in response or "yes" in response

    
def answer_question(cur_state, question):

    with open('system_prompts/android_data_collection/question_answer.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
UI: 
{}
Question: {}
""".format(cur_state, question)
    response = call_llm(prompt, system_prompt)
    print(response)
    try:
        explanation, answer = [l for l in response.split('\n') if l != '']
        explanation = explanation.split("Explanation: ")[1].strip()
        answer = answer.split("Answer: ")[1].strip()
    except:
        i1 = response.find("Explanation:")
        i2 = response.find("Answer:")
        explanation = response[i1 + len("Explanation:"):i2].strip()
        answer = response[i2 + len("Answer:"):].strip()
    explanation = explanation + f" The action I'll take is `stop [{answer}]`."
    return explanation, answer

def pre_reasoning(high_level_intent, step_history):
    with open('system_prompts/android_data_collection/pre-reasoning_summarize.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """Input:
High Level intent: {}
Step history: {}""".format(high_level_intent, step_history)
    response = call_llm(prompt, system_prompt)
    pre_reasoning, instruction = response.split("Instruction:")
    pre_reasoning = pre_reasoning.split("Thought:")
    print("pre reasoning:", pre_reasoning)
    print("paraphrase Instruction:", instruction)
    return instruction


############################################
###### Thought related functions ######
############################################
def analysis_thought(state, step_history):
    with open('system_prompts/android_data_collection/page_analysis.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
Page: 
{}
Previous steps: 
{}

Thought: Let's think step by step. """.format(state, step_history)
    response = call_llm(prompt, system_prompt)
    return response

def rephrase_thought(intent, thought_history, action_history):
    with open('system_prompts/android_data_collection/rephrase_thought.txt', 'r') as f:
        sys_prompt = f.read()
    prompt = '''
Original thoughts: 
{}

Goal: {}
Actions: {}

Rewritten thoughts:
'''.format(thought_history, intent, ', '.join(action_history))
    response = call_llm(prompt, sys_prompt)
    try:
        thought_list = ast.literal_eval(response)
    except:
        if not response.startswith('['):
            thought_list = [r for r in response.split('\n') if r != '']
        else:
            thought_list = [r for r in response.split('\n')[1: -1] if r != '']
        for t in thought_list:
            t = t.strip()
    return thought_list

def replace_thought(trajectory, new_thought_traj):
    tmp_traj_copy = []
    for i in range(len(trajectory)):
        new_tuple = (trajectory[i][0], new_thought_traj[i].replace(f"Thought {i+1}: ", ''), trajectory[i][2], trajectory[i][3], trajectory[i][4])
        tmp_traj_copy.append(new_tuple)
    return tmp_traj_copy

def align_thought_action(action_traj, new_thought_traj):
    # return True if not discarded, and False otherwise
    for a, t in zip(action_traj, new_thought_traj):
        if a not in t:
            return False
    return True


############################################
###### Main exploration functions ######
############################################
def trajectory_eval(instruction, step_history):
    with open('system_prompts/android_data_collection/trajectory_evaluation.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
Instruction:
{}
Steps:
{}
""".format(instruction, '\n'.join(step_history))
    response = call_llm(prompt, system_prompt, model='gpt-4o-mini')
    return "Yes" in response or "yes" in response

def task_summarize(step_history, general_flag):
    # trajectory summarize intent
    if not general_flag:
        with open('system_prompts/android_data_collection/summarize_prompt.txt', 'r') as f:
            system_prompt = f.read()
    else:
        with open('system_prompts/android_data_collection/summarize_prompt_general.txt', 'r') as f:
            system_prompt = f.read()
    prompt = """
Input:
Previous steps: {}

Thought:
    """.format(step_history)
    response = call_llm(prompt, system_prompt, model='gpt-4o-mini')
    print("high level task:", response)
    try:
        high_level_task = response.split("Task: ")[1]
    except:
        # error format, extract the last line
        lines = response.splitlines()
        high_level_task = next(line for line in reversed(lines) if line.strip())
    
    return high_level_task.strip()

def thought_action_gen(state: str, guide, step_history, early_stop=False):

    with open('system_prompts/android_data_collection/act_prompt.txt', 'r') as f:
        sys_prompt = f.read()
    avail_actions = actions.AVAILABLE_ACTIONS.values()

    sys_prompt = sys_prompt.format(avail_actions)

    # add index to each element in the state
    indexed_state = ''
    for i, line in enumerate(state.strip().split('\n')):
        indexed_state += f"Element {i}: {line}\n"

    
    prompt = '''
Input:
Guide: {}
Current state: {}
Previous steps: {}\nIf you want to open an APP, just call the `open_app` action. Do not do any other action.'''.format(guide, indexed_state, step_history)
    if early_stop:
        stop_guide = """**If you think the guide has been completed, and you want to finish the current browsing process, put "stop []" in "Action" field, or put "stop [unachievable]" if you think the task cannot be completed, and None in "Task" field.**\n"""
        prompt += stop_guide

    response = call_llm(prompt, sys_prompt, model='gpt-4o-mini')
    print(response + "\n")
    
    action, thought, task = "", "", ""
    thought = response[response.find("Thought: "):response.find("Action: ")].split("Thought: ")[1].strip()
    action = response[response.find("Action: "):response.find("Task: ")].split("Action: ")[1].strip()
    task = response[response.find("Task: "):].split("Task: ")[1].strip()
    
    return thought, action, task


@timer
def android_sim_traj(init_state: str,
                      guide, 
                      fixed_guides: str = None,
                      num_guides_per_step=5,
                      min_step=3, 
                      max_step=5, 
                      index=1, 
                      num=1, 
                      save=True,
                      rag_enabled=False):
    if rag_enabled:
        print("####RAG enabled####")
        # extract the current app name from the single quote from the guide
        match = re.search(r"'([^']+)'", guide)
        if match:
            app_name = match.group(1)
            simulator = RAG_AndroidSimulator(domain=app_name)
        else:
            raise ValueError("No app name found in the guide.")
    else:
        print("####RAG disabled####")
        simulator = AndroidSimulator()
    simulator.reset(init_state)
    trajectory = []
    thought_action_traj = []
    step_history = []
    action_history = []
    guides = []
    i = 1

    retry = 0
    allowed_modify_time = min_step - 1
    early_stop_flag = False
    early_stop_thought = ""
    completion = False
    general_flag = False
        
    if fixed_guides:
        fixed_guides = fixed_guides.split('.')
        fixed_guides = replace_entity(fixed_guides)
        min_step = len(fixed_guides)
        max_step = min_step + 4
        guide = fixed_guides[0].strip()
        fixed_guides.pop(0)
        breakpoint()
        
    
    print("********" + str(guide) + "********")
    guides.append(guide)
    # set the baseline guide as general guide

    print(os.environ['OPENAI_API_KEY'])
    print(os.environ['OPENAI_ORG_ID'])
    while i <= max_step:
        # stop the trajectory if the guide is completed
        if completion:
            break
        # stop the trajectory if the trajectory retried too many times
        if retry > 3:
            print("retry too many times, stop the trajectory")
            return None
        
        # get the whole state
        state = simulator.get_str_backup_current_state()
        print(f"############# index {index}, iter {num}, step {i} #############")
        error = False
        early_stop = i > min_step
        # generate thought and action at current step
        print("guide for thought action generation:", guide)
        try:
            thought, action, task = thought_action_gen(state, guide, step_history, early_stop)
        except Exception as e:
            print("Unexpected error during thought action generation:", e)
            error = True
            retry += 1
            continue
        if str(action) not in thought:
            thought = thought + " In summary, the next action I will perform is " + str(action)

        if i >= min_step and "stop" in action:
            early_stop_flag = True
            print("###### final thought ######")
            early_stop_thought = thought
            print(thought)
            print("############")
            break

        # correct the action if it's not in the action space
        try:
            action = match_action(action)  # class
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("original action:", action)
            error = True
            retry += 1
            continue

        # post process the action
        cur_actions = []
        if isinstance(action, Click) or isinstance(action, Type):
            # print("**********************")
            # print("state for find element")
            # print(simulator._get_backup_current_state())
            # print("**********************")
            target_element = simulator.find_element_by_id(simulator._get_backup_current_state(), action.id)
            # print(target_element)

            if target_element is None:
                error = True
                retry += 1
                print("target element not found")
                continue

            coord = target_element.bbox_pixels
            if (coord[2] + simulator.coord_offset[simulator.window_index][1]) >= simulator.height:
                scroll_time = int((coord[2] + simulator.coord_offset[simulator.window_index][1] - simulator.height) // simulator.height + 1)
                cur_actions += [Scroll('down')] * scroll_time
                print("scroll down", scroll_time)
            elif (coord[3] + simulator.coord_offset[simulator.window_index][1]) < 0:
                scroll_time = int(abs((coord[3] + simulator.coord_offset[simulator.window_index][1]) // simulator.height) + 1)
                cur_actions += [Scroll('up')] * scroll_time
                print("scroll up", scroll_time)
        cur_actions.append(action)

        # take the step
        if len(cur_actions) >= 4:
            # discard the action if the action is too long
            retry += 1
            continue

        print("actions with scrolls")
        for action in cur_actions:
            print(str(action))

        # execute the actions
        for action in cur_actions:
            prev_state = copy.deepcopy(simulator.get_str_cur_state())
            new_state = None

            try:
                if isinstance(action, Scroll):
                    analysis = analysis_thought(simulator.get_str_cur_state(), step_history)
                    step_thought = "Let's think step by step. " + analysis + f"I think the content I want is not appearing in current window, but it should be on the current webpage. So I'll scroll down to find more information. In summary, the next action I will perform is scroll [{action.direction}]"
                    step_task = f"Scroll {action.direction} the current page to find more information on the page."
                else:
                    step_thought = thought
                    step_task = task

                # RAG: simulate next state with action history
                if rag_enabled:
                    new_state = simulator.step(action, action_history + [step_task])
                else:
                    if len(cur_actions) > 1 and not isinstance(action, Scroll):
                        # modify the index of the action
                        target_element = simulator.find_element_by_id(simulator._get_backup_current_state(), action.id)
                        for i, e in enumerate(simulator.cur_state):
                            if e == target_element:
                                action.id = i + 1
                                break

                    new_state = simulator.step(action)

                # print("*************")
                # print(a11y_tree_state)
                # print("*************")
            except Exception as e:
                print("Unexpected Error:", e)
                print(traceback.format_exc())
                # wrong generated next state, but action taken, cannot revert.
                error = True
                new_state = None
                retry += 1
            if new_state == prev_state:
                print("Page doesn't change")
                error = True
                retry += 1

            if error:
                break
                
            # record the thought and action
            str_action = str(action)
            thought_action = {
                f"Thought": step_thought,
                f"Action": str(action)
            }

            history = copy.deepcopy(step_history)
            tmp_traj = (str(prev_state), step_thought, str_action, step_task, history)
            thought_action_traj.append(thought_action)
            step_history.append(step_task)
            
            # print(i, history)
            trajectory.append(tmp_traj)

            if not isinstance(action, WA_Scroll):
                # check if general guide is completed
                completion = judge_guide_completion(guide, step_history, simulator.get_str_cur_state())
                print("check guide completion:", completion)
                
                # generate task guide if guide is completed
                if allowed_modify_time > 0 and completion and i <= 2 * min_step:
                    
                    # generate task guide if guide is completed
                    if completion:
                        if fixed_guides:
                            guide = fixed_guides[0].strip()
                            fixed_guides.pop(0)
                        else:
                            tmp = task_guidance(simulator.get_str_cur_state(), guide, step_history, first_guide=False, num_guides=num_guides_per_step)
                            print("next step guides")
                            print(tmp)
                            if len(tmp) > 0:
                                guide = random.sample(tmp, 1)[0]

                            # check if guide is search related
                            if i == 1:
                                result = judge_missed_entity(guide)
                                
                                if result:
                                    guide = specify_entity(guide)
                        completion = False
                        guides.append(guide)

                    allowed_modify_time -= 1

                elif completion:
                    # stop the trajectory
                    break

        if not error:
            i += 1
    
    # adding the final step to the trajectory
    if early_stop_flag:
        stop_thought = early_stop_thought
    else:
        analysis = analysis_thought(simulator.get_str_cur_state(), step_history)
        # print("###### analysis ######")
        # print(analysis)
        # print("############")
        stop_thought = f"Let's think step by step. {analysis} I think I've completed the task. The action I'll take is stop []."

    stop_action = 'stop []' if 'unachievable' not in str_action else 'stop [unachievable]'

    terminal_state = simulator.get_str_cur_state()
    step_tuple = (terminal_state, stop_thought, stop_action, "stop", step_history)
    trajectory.append(step_tuple)
    thought_action_traj.append({"Thought": stop_thought, "Action": stop_action})

    # summarize the high-level intent, based on the trajectory
    tmp_step_history = copy.deepcopy(step_history)
    # adding actions as prefix to the step history
    for t, s in zip(thought_action_traj, tmp_step_history):
        s = t['Action'] + ": " + s

    high_level_intent = task_summarize(tmp_step_history, general_flag)
    thought_traj = [f"Thought {i+1}: " + d['Thought'] for i, d in enumerate(thought_action_traj)]
    action_traj = [d['Action'] for d in thought_action_traj]
    
    # save un-rewritten trajectory
    if save:
        # check trajectory quality
        # result = trajectory_eval(high_level_intent, step_history)
        # if not result:
        #     return None

        # rewrite the thought according to the high-level intent
        new_thought_traj = rephrase_thought(high_level_intent, thought_traj, action_traj)

        # DOUBLE CHECK if rephrasing thought also change the original action
        keep_traj = align_thought_action(action_traj, new_thought_traj)
        if not keep_traj:
            print("action doesn't match thought")
            return None

        # print("###### rephrase thought ######")
        # print(new_thought_traj[-1])
        # print("############")
        if fixed_guides is not None:
            directory = f'train_set_android_balanced/init_state_{index}_maxlen={max_step}_new/iter_{num}'
        elif rag_enabled:
            directory = f'train_set_android_rag/init_state_{index}_maxlen={max_step}/iter_{num}'
        else:
            directory = f'train_set_android/init_state_{index}_maxlen={max_step}/iter_{num}'
        os.makedirs(directory, exist_ok=True)

        with open(f'{directory}/original_thoughts.txt', 'w') as f:
            for t in thought_traj:
                f.write(t + "\n")
        with open(f'{directory}/thoughts.txt', 'w') as f:
            for t in new_thought_traj:
                f.write(t + "\n")
        with open(f'{directory}/guides.txt', 'w') as f:
            f.write('\n'.join(guides))
        with open(f'{directory}/actions.txt', 'w') as f:
            for k in range(len(trajectory)):
                f.write(trajectory[k][2] + "\n")
        with open(f'{directory}/instruction.txt', 'w') as f:
            f.write(high_level_intent)
        new_traj = replace_thought(trajectory, new_thought_traj)
        # use pickle to save the trajectory
        with open(f'{directory}/trajectory.pkl', 'wb') as f:
            pickle.dump(new_traj, f)


    # add reasoning step
    try:
        if not general_flag:
            traj_answer = None

            # 0. stripping the scroll actions at the end
            while 'scroll' in action_traj[-1]:
                thought_traj.pop()
                action_traj.pop()
                thought_action_traj.pop()
                trajectory.pop()
                step_history.pop()

            # 1. check whether a reasoning step is needed
            do_reasoning = judge_reasoning(high_level_intent, step_history)

            if do_reasoning:
                # 2. generate reasoning tasks based on the current state
                
                terminal_state = simulator.get_str_cur_state()
                # add index to each element in the state
                indexed_state = ''
                for i, line in enumerate(terminal_state.strip().split('\n')):
                    indexed_state += f"Element {i}: {line}\n"
                post_reasoning_tasks = propose_reasoning_tasks(indexed_state)
                print("reasoning_task: ", post_reasoning_tasks)

                for i, post_reasoning_task in enumerate(post_reasoning_tasks):
                    explanation, answer = answer_question(indexed_state, post_reasoning_task)
            
                    if 'None' not in answer:
                        post_reasoning_instruction = post_reasoning_task.strip()
                        traj_answer = answer.strip()
                        if save:
                            new_thought_traj = rephrase_thought(post_reasoning_instruction, thought_traj, action_traj)

                            # DOUBLE CHECK if rephrasing thought also change the original action
                            keep_traj = align_thought_action(action_traj, new_thought_traj)
                            if not keep_traj:
                                print("action doesn't match thought")
                                continue

                            new_traj = replace_thought(trajectory, new_thought_traj)
                            analysis = analysis_thought(simulator.get_str_cur_state(), step_history)

                            new_traj[-1] = (terminal_state, f"Let's think step by step. {analysis} I think I'm ready to answer the question: {post_reasoning_instruction}. {explanation}", f"stop [{traj_answer}]", "stop", new_traj[-1][4])
                            thought_action_traj[-1]['Action'] = f"stop [{traj_answer}]"

                            if fixed_guides is not None:
                                directory = f'train_set_android_balanced/init_state_{index}_maxlen={max_step}_new/iter_{num}_reasoning_{i}'
                            elif rag_enabled:
                                directory = f'train_set_android_rag/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
                            else:
                                directory = f'train_set_android/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
                            os.makedirs(directory, exist_ok=True)

                            with open(f'{directory}/actions.txt', 'w') as f:
                                for k in range(len(new_traj)):
                                    f.write(new_traj[k][2] + "\n")
                            with open(f'{directory}/instruction.txt', 'w') as f:
                                f.write(post_reasoning_instruction)
                            with open(f'{directory}/answer.txt', 'w') as f:
                                f.write(traj_answer)
                            with open(f'{directory}/trajectory.pkl', 'wb') as f:
                                pickle.dump(new_traj, f)

    except Exception as e:
        print(e)
    # try:
    #     pre_reasoning_task = pre_reasoning(high_level_intent, step_history)
    #     if 'None' not in pre_reasoning_task:
    #         high_level_intent = pre_reasoning_task
    # except Exception as e:
    #     print(e)
    return high_level_intent


############################################
###### Data collection functions ######
############################################
def process_sim_collect(init_state, guides, num_guides_per_step, min_step=3, max_step=5, index=1, save=True, fixed_guides=False, rag_enabled=False):
    # print(guides)
    
    num = len(guides)
    if num == 0:
        return None
    if not fixed_guides:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(
                android_sim_traj,
                [init_state] * num,
                guides,
                [None] * num,
                [num_guides_per_step] * num,
                [min_step] * num,
                [max_step] * num,
                [index] * num,
                [i for i in range(num)],
                [save] * num,
                [rag_enabled] * num
            )
    # else:
    #     with ThreadPoolExecutor(max_workers=num // 2 + 1) as executor:
    #         results = executor.map(
    #             webarena_sim_traj,
    #             [init_state] * num,
    #             [''] * num,
    #             guides,
    #             [min_step] * num,
    #             [max_step] * num,
    #             [index] * num,
    #             [i for i in range(num)],
    #             [save] * num
    #         )
    return results
        
def collect(init_states, num_guides_per_step, min_length_range: list, nums: list, debug=False, rag_enabled=True):

    if debug:
        min_step = min_length_range[0]
        for state in init_states[:1]:
            simulator = AndroidSimulator()
            try:
                simulator.reset(state)
            except Exception as e:
                print("Unexpected Error:", e)
                init_states.remove(state)
                continue

            tasks = task_guidance(state, num_guides = num_guides_per_step)
            print(tasks)

        guide = "Open the app 'Camera'"
        for i in range(nums[0]):
            android_sim_traj(init_states[0], guide, None, num_guides_per_step, min_step=min_step, max_step=min_step+6, index=1000, num=i, save=True, rag_enabled=rag_enabled)

    else:
        # Propose first-step guides
        guides = []
        filtered_init_states = []
        for state in init_states:
            simulator = AndroidSimulator()
            try:
                simulator.reset(state)
            except Exception as e:
                print("Unexpected Error:", e)
                continue

            tasks = task_guidance(state, num_guides = num_guides_per_step)
            print(tasks)
            guides.append(tasks)
            filtered_init_states.append(state)

        # collect first-round trajectories
        for l, num in zip(min_length_range, nums):
            sampled_guides = []
            for p, g in zip(filtered_init_states, guides):
                
                sample_guide = (num // len(g)) * g + g[:num % len(g)]
                # print(sample_guide)
                sampled_guides.append(sample_guide)
            
            with ProcessPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    process_sim_collect,
                    filtered_init_states,
                    sampled_guides,
                    [num_guides_per_step] * len(filtered_init_states),
                    [l] * len(filtered_init_states),
                    [l+6] * len(filtered_init_states),
                    [i for i in range(len(filtered_init_states))],
                    [True] * len(filtered_init_states),
                    [False] * len(filtered_init_states),
                    [rag_enabled] * len(filtered_init_states)
                )


def get_init_states():
    root = f'init_states/android'
    init_states = []
    files = os.listdir(root)
    for file in files:
        with open(f'{root}/{file}', 'r') as f:
            state = f.read()
            init_states.append(state)
    return init_states

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rag_enabled', action='store_true')
    parser.add_argument('--min_steps', type=int, default=1, nargs='+')
    parser.add_argument('--nums', type=int, default=10, nargs='+')
    parser.add_argument('--num_guides_per_step', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if isinstance(args.min_steps, int):
        args.min_steps = [args.min_steps]
    if isinstance(args.nums, int):
        args.nums = [args.nums]
    init_states = get_init_states()

    collect(init_states, args.num_guides_per_step, args.min_steps, args.nums, debug=args.debug, rag_enabled=args.rag_enabled)
