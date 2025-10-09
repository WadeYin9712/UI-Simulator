import random
import actions
from actions.action import *
import copy
import os
import ast
import pickle
import re

from utils import *
import traceback
import argparse

from android_world.env import env_launcher
from android_world.env import json_action
from android_world.agents import m3a_utils
from android_world.env import representation_utils
from android_world.agents import agent_utils

def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )

def _generate_ui_elements_description_list_full(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info

def _parse_action(action: str, index_map: dict = None) -> dict:
    # parse the action to the expected format
    action = action.strip()

    if action.startswith("click"):
        match = re.match(r"click\s*\[(.*?)\]", action)
        if match:
            target_index = int(match.group(1))
            if index_map is not None:
                target_index = index_map.get(target_index, target_index)
            return {"action_type": "click", "index": target_index}

    elif action.startswith("input_text"):
        match = re.match(r"input_text\s*\[(.*?)\]\s*\[(.*?)\]", action)
        if match:
            target_index, text_input = int(match.group(1)), match.group(2)
            if index_map is not None:
                target_index = index_map.get(target_index, target_index)
            return {"action_type": "input_text", "text": text_input, "index": target_index}

    elif action.startswith("open_app"):
        match = re.match(r"open_app\s*\[(.*?)\]", action)
        if match:
            app_name = match.group(1)
            return {"action_type": "open_app", "app_name": app_name}

    elif action == "keyborad_enter":
        return {"action_type": "keyboard_enter"}

    elif action.startswith("scroll"):
        match = re.match(r"scroll\s*\[(.*?)\]", action)
        if match:
            direction = match.group(1)
            return {"action_type": "scroll", "direction": direction}

    elif action == "navigate_back":
        return {"action_type": "navigate_back"}

    elif action == "navigate_home":
        return {"action_type": "navigate_home"}

    elif action == "wait":
        return {"action_type": "wait"}

    elif action.startswith("stop"):
        match = re.match(r"stop\s*\[(.*?)\]", action)
        if match:
            answer = match.group(1)
            if answer == "":
                return {"action_type": "status", "goal_status": "complete"}
            elif answer == 'N/A':
                return {"action_type": "status", "goal_status": "infeasible"}
            else:
                return {"action_type": "answer", "text": answer}

    else:
        raise ValueError(f"Unknown action format: {action}")

############################################
###### Guide related functions ######
############################################

def judge_missed_entity(guide):
    # check if guide is search task or not
    with open('../system_prompts/android_data_collection/judge_entity_guide.txt', 'r') as f:
        sys_prompt = f.read()
    prompt = '''
Original task: {}'''.format(guide)
    response = call_llm(prompt, sys_prompt)
    return "Yes" in response or "yes" in response

def specify_entity(guide):
    # print(description)
    # specific entity
    with open('../system_prompts/android_data_collection/specific_entity.txt', 'r') as f:
        sys_prompt = f.read()
    prompt = '''
Original task:
{}
Thought: I'll try to'''.format(guide)
    response = call_llm(prompt, sys_prompt)
    guide = re.split(r'(?i)tasks:', response)[1].strip().split('\n')
    guide = [line.split('. ', 1)[1].strip() for line in guide if line and line[0].isdigit()]
    # print(guide)
    guide = random.sample(guide, 1)[0]
    return guide

def replace_entity(guides: list):
    with open('../system_prompts/android_data_collection/replace_entity.txt', 'r') as f:
        system_prompt = f.read()
    prompt = '''
Guides:
{}

New guides:\n'''.format('\n'.join(guides))
    response = call_llm(prompt, system_prompt)
    guides = response.split('\n')

    return guides

def judge_guide_completion(guide, prev_steps, cur_state):
    with open('../system_prompts/android_data_collection/judge_guide_completion.txt', 'r') as f:
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
        tasks = [
            "Open the app 'Settings'",
            "Open the app 'Clock'",
            "Open the app 'Audio Recorder'",
            "Open the app 'Broccoli', an app for recipes management",
            "Open the app 'Simple Calendar Pro'",
            "Open the app 'Camera'",
            "Open the app 'Contacts'",
            "Open the app 'Simple Draw Pro'",
            "Open the app 'Files'",
            "Open the app 'Simple Gallery Pro'",
            "Open the app 'Joplin'",
            "Open the app 'Markor', a free, open-source Markdown text editor",
            "Open the app 'OpenTracks'",
            "Open the app 'OsmAnd', a map app",
            "Open the app 'Pro Expense', a professional expense tracking app",
            "Open the app 'Retro Music'",
            "Open the app 'Simple SMS Messenger'",
            "Open the app 'Tasks'",
            "Open the app 'VLC'",
        ]
        random.shuffle(tasks)
        return tasks

    else:
        with open('../system_prompts/android_data_collection/task_guide.txt', 'r') as f:
            sys_prompt = f.read()

        example_guides = [
            "1. Create a new contact'"
            "2. Switch to 'Phone contacts'"
            "3. Open the navigation drawer."
        ]
        example_guides = '\n'.join(example_guides[:num_guides])
        sys_prompt = sys_prompt.format(num_guides, example_guides)

        # TODO: clean unused element attributes or not?
        # invariant_elements, new_elements = split_state(prev_state, cur_state)

        prompt = '''
Elements:
{}

Original task guide:
{}
Previous steps:
{}

'''.format(cur_state, prev_guide, prev_step)
        prompt += "Thought: Let's think step by step."
    
    response = call_llm(prompt, sys_prompt)
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
    with open('../system_prompts/android_data_collection/reasoning_guide.txt', 'r') as f:
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
    with open('../system_prompts/android_data_collection/judge_reasoning.txt', 'r') as f:
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

    with open('../system_prompts/android_data_collection/question_answer.txt', 'r') as f:
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
    with open('../system_prompts/android_data_collection/pre-reasoning_summarize.txt', 'r') as f:
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
    with open('../system_prompts/android_data_collection/page_analysis.txt', 'r') as f:
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
    with open('../system_prompts/android_data_collection/rephrase_thought.txt', 'r') as f:
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
    with open('../system_prompts/android_data_collection/trajectory_evaluation.txt', 'r') as f:
        system_prompt = f.read()
    prompt = """
Instruction:
{}
Steps:
{}
""".format(instruction, '\n'.join(step_history))
    response = call_llm(prompt, system_prompt)
    return "Yes" in response or "yes" in response

def task_summarize(step_history, general_flag):
    # trajectory summarize intent
    if not general_flag:
        with open('../system_prompts/android_data_collection/summarize_prompt.txt', 'r') as f:
            system_prompt = f.read()
    else:
        with open('../system_prompts/android_data_collection/summarize_prompt_general.txt', 'r') as f:
            system_prompt = f.read()
    prompt = """
Input:
Previous steps: {}

Thought:
    """.format(step_history)
    response = call_llm(prompt, system_prompt)
    print("high level task:", response)
    try:
        high_level_task = response.split("Task: ")[1]
    except:
        # error format, extract the last line
        lines = response.splitlines()
        high_level_task = next(line for line in reversed(lines) if line.strip())
    
    return high_level_task.strip()

def thought_action_gen(state: str, guide, step_history, early_stop=False):

    with open('../system_prompts/android_data_collection/act_prompt.txt', 'r') as f:
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

    response = call_llm(prompt, sys_prompt)
    print(response + "\n")
    
    action, thought, task = "", "", ""
    thought = response[response.find("Thought: "):response.find("Action: ")].split("Thought: ")[1].strip()
    action = response[response.find("Action: "):response.find("Task: ")].split("Action: ")[1].strip()
    task = response[response.find("Task: "):].split("Task: ")[1].strip()
    
    return thought, action, task


@timer
def android_real_traj(guide, 
                      num_guides_per_step=5,
                      min_step=3, 
                      max_step=5, 
                      index=1,
                      num=1,
                      save=True):

    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        adb_path=_find_adb_directory(),
    )

    env.reset(go_home=True)
    env.hide_automation_ui()
    
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
        env_state = env.get_state(wait_to_stabilize=False)
        ui_elements = env_state.ui_elements
        state = _generate_ui_elements_description_list_full(
            ui_elements,
            env.logical_screen_size
        )

        print(f"############# index {index}, step {i} #############")
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

        # post process the action
        cur_actions = []

        cur_actions.append(action)

        print("actions with scrolls")
        for action in cur_actions:
            print(str(action))

        # execute the actions
        for action in cur_actions:
            prev_state = str(state)
            new_state = None

            try:
                if isinstance(action, Scroll):
                    analysis = analysis_thought(prev_state, step_history)
                    step_thought = "Let's think step by step. " + analysis + f"I think the content I want is not appearing in current window, but it should be on the current webpage. So I'll scroll down to find more information. In summary, the next action I will perform is scroll [{action.direction}]"
                    step_task = f"Scroll {action.direction} the current page to find more information on the page."
                else:
                    step_thought = thought
                    step_task = task

                parsed_action = str(_parse_action(action))
                converted_action = json_action.JSONAction(
                    **agent_utils.extract_json(parsed_action),
                )
                env.execute_action(converted_action)

                new_env_state = env.get_state(wait_to_stabilize=True)
                new_ui_elements = new_env_state.ui_elements
                new_state = _generate_ui_elements_description_list_full(
                    new_ui_elements,
                    env.logical_screen_size
                )

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

            if not isinstance(action, Scroll):
                # check if general guide is completed
                completion = judge_guide_completion(guide, step_history, new_state)
                print("check guide completion:", completion)
                
                # generate task guide if guide is completed
                if allowed_modify_time > 0 and completion and i <= 2 * min_step:
                    
                    # generate task guide if guide is completed
                    if completion:
                        tmp = task_guidance(new_state, guide, step_history, first_guide=False, num_guides=num_guides_per_step)
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
        analysis = analysis_thought(new_state, step_history)
        # print("###### analysis ######")
        # print(analysis)
        # print("############")
        stop_thought = f"Let's think step by step. {analysis} I think I've completed the task. The action I'll take is stop []."

    stop_action = 'stop []' if 'unachievable' not in str_action else 'stop [unachievable]'

    terminal_state = new_state
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
        directory = f'train_set_android_real/init_state_{index}_maxlen={max_step}/iter_{num}'
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
                
                terminal_state = new_state
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
                            analysis = analysis_thought(terminal_state, step_history)

                            new_traj[-1] = (terminal_state, f"Let's think step by step. {analysis} I think I'm ready to answer the question: {post_reasoning_instruction}. {explanation}", f"stop [{traj_answer}]", "stop", new_traj[-1][4])
                            thought_action_traj[-1]['Action'] = f"stop [{traj_answer}]"

                            directory = f'train_set_android_real/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
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

    env.close()
    return high_level_intent


############################################
###### Data collection functions ######
############################################
        
def collect_real_traj(init_states, num_guides_per_step, min_length_range: list, nums: list, debug=False, rag_enabled=True):

    if debug:
        min_step = min_length_range[0]
        for state in init_states[:1]:

            tasks = task_guidance(state, num_guides = num_guides_per_step)
            print(tasks)

        guide = "Open the app 'Contacts'"
        for i in range(nums[0]):
            android_real_traj(guide, num_guides_per_step, min_step=min_step, max_step=min_step+6, index=1000, num=i, save=True)

        # fixed_guide = 'Search for Sony WH-1000XM4. Add Sony WH-1000XM4 to cart. Proceed to checkout.'
        # for i in range(num):
        #     webarena_sim_traj(init_states[0], guide='', fixed_guides=fixed_guide, min_step=min_step, max_step=min_step+4, index=1000, num=i, save=True, webarena_obs_format=True)
    else:
        # Propose first-step guides
        guides = []
        filtered_init_states = []
        for state in init_states:

            tasks = task_guidance(state, num_guides = num_guides_per_step)
            print(tasks)
            random.shuffle(tasks)
            guides.append(tasks)
            filtered_init_states.append(state)

        # collect first-round trajectories
        for l, num in zip(min_length_range, nums):
            for id, (p, g) in enumerate(zip(filtered_init_states, guides)):
                sample_guide = (num // len(g)) * g + g[:num % len(g)]
                # print(sample_guide)

                for i, guide in enumerate(sample_guide):
                    try:
                        android_real_traj(
                            guide,
                            num_guides_per_step,
                            min_step=l,
                            max_step=l+6,
                            index=id,
                            num=i,
                            save=True
                        )
                    except Exception as e:
                        print("Unexpected Error during trajectory collection:", e)
                        print(traceback.format_exc())
                        continue

def get_init_states():
    root = f'../init_states/android'
    # state_paths = ['example_27.json']
    init_states = []
    files = os.listdir(root)
    for file in files:
        if file.startswith('home'):
            with open(f'{root}/{file}', 'r') as f:
                state = f.read()
                init_states.append(state)
    return init_states

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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

    collect_real_traj(init_states, args.num_guides_per_step, args.min_steps, args.nums, debug=args.debug)
