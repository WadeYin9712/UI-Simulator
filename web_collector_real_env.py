"""
Code for collecting trajectories in real webarena environment
with step-wise guided rollout process

"""

import random
import actions
import copy
import os
import pickle
import subprocess
from web_collector import *
from webarena.browser_env import ScriptBrowserEnv, create_id_based_action

def prepare_webarena_env(file_idx):
    config_file = f'webarena/config_files/{file_idx}.json'
    env = ScriptBrowserEnv(
        headless=True,
        slow_mo = 0,
        observation_type='accessibility_tree',
        current_viewport_only=True,
        viewport_size={
            'width': 1280,
            'height': 720
        },
        save_trace_enabled=False,
        sleep_after_execution=0.0
    )

    obs, info = env.reset(options={'config_file': config_file})
    state_info = {'observation': obs, 'info': info}
    return env, state_info

def webarena_real_traj(env, 
                       init_state: dict,
                       domain, 
                       guide: str,
                       num_guides_per_step=5,
                       min_step=3,
                       max_step=5,
                       index=1,
                       num=1,
                       save=True,
                       general_searching=0.5):
    state, _ = init_state['observation']['text'], init_state['info']
    state = state.split('\n\n')[1]

    trajectory = []
    thought_action_traj = []
    step_history = []
    guides = []
    i = 1

    retry = 0
    allowed_modify_time = min_step - 1
    early_stop_flag = False
    early_stop_thought = ""
    completion = False
    general_flag = False

    available_actions = actions.AVAILABLE_WEBARENA_ACTIONS
    available_actions['scroll'] = WA_Scroll.INTRO

    # check if guide is search related
    result = judge_missed_entity(guide)
    
    if result:
        if random.random() < general_searching:
            guide = general_entity(state, guide)
            general_flag = True
        else:
            guide = specify_entity(state, guide)
                
    print("********" + str(guide) + "********")
    guides.append(guide)
    # set the baseline guide as general guide


    while i <= max_step:
        # stop the trajectory if the guide is completed
        if completion:
            break
        # stop the trajectory if the trajectory retried too many times
        if retry > 3:
            print("retry too many times, stop the trajectory")
            return None
        
        
        print(f"############# index {index}, iter {num}, step {i} #############")
        error = False
        early_stop = i > min_step
        # generate thought and action at current step
        print("guide for thought action generation:", guide)
        # ensure that we have scroll action in the action space
        thought, action, task = thought_action_gen(domain, state, guide, step_history, early_stop)
        if str(action) not in thought:
            thought = thought + " In summary, the next action I will perform is " + str(action)

        if i >= min_step and "stop" in action:
            early_stop_flag = True
            print("###### final thought ######")
            early_stop_thought = thought
            print(thought)
            print("############")
            break

        prev_state = str(state)
        try:
            step_thought = thought
            step_task = task

            webarena_action = create_id_based_action(action)
            state, _, terminated, _, info = env.step(webarena_action)
            state = state['text'].split('\n\n')[1]

            if terminated:
                break
            
        except Exception as e:
            print("Unexpected Error:", e)
            # wrong generated next state, but action taken, cannot revert.
            error = True
            retry += 1

        if state.strip() == prev_state.strip():
            print("Page doesn't change")
            error = True
            retry += 1

        if error:
            continue
            
        # record the thought and action
        webarena_action = str(action)
        thought_action = {
            f"Thought": step_thought,
            f"Action": str(action)
        }

        history = copy.deepcopy(step_history)
        tmp_traj = (prev_state, step_thought, webarena_action, step_task, history)
        thought_action_traj.append(thought_action)
        step_history.append(step_task)
        
        # print(i, history)
        trajectory.append(tmp_traj)

        # check if general guide is completed
        completion = judge_guide_completion(guide, step_history, prev_state, state)
        print("check guide completion:", completion)
        
        # generate task guide if guide is completed
        if allowed_modify_time > 0 and completion and i <= 2 * min_step:
            
            # generate task guide if guide is completed
            if completion:
                tmp = task_guidance(state, guide, step_history, first_guide=False, num_guides=num_guides_per_step, domain=domain)
                print("next step guides")
                print(tmp)
                if len(tmp) > 0:
                    guide = random.sample(tmp, 1)[0]
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
        analysis = analysis_thought(state, step_history)
        # print("###### analysis ######")
        # print(analysis)
        # print("############")
        stop_thought = f"Let's think step by step. {analysis} I think I've completed the task. The action I'll take is stop []."

    stop_action = 'stop []' if 'unachievable' not in webarena_action else 'stop [unachievable]'

    terminal_state = state
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
        #     print("trajectory not qualified")
        #     return None

        # rewrite the thought according to the high-level intent
        new_thought_traj = rephrase_thought(high_level_intent, thought_traj, action_traj)
        
        # DOUBLE CHECK if rephrasing thought also change the original action
        keep_traj = align_thought_action(action_traj, new_thought_traj)
        if not keep_traj:
            print("action doesn't match thought")
            # breakpoint()
            return None

        # print("###### rephrase thought ######")
        # print(new_thought_traj[-1])
        # print("############")
        directory = f'train_set_real/init_state_{index}_maxlen={max_step}/iter_{num}'
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

            # 1. check whether a reasoning step is needed
            do_reasoning = judge_reasoning(high_level_intent, step_history)

            if do_reasoning:
                # 2. generate reasoning tasks based on the current state
                
                post_reasoning_tasks = propose_reasoning_tasks(terminal_state)
                print("reasoning_task: ", post_reasoning_tasks)

                for i, post_reasoning_task in enumerate(post_reasoning_tasks):
                    explanation, answer = answer_question(terminal_state, post_reasoning_task)
            
                    if 'None' not in answer:
                        post_reasoning_instruction = post_reasoning_task.strip()
                        traj_answer = answer.strip()
                        if save:
                            new_thought_traj = rephrase_thought(post_reasoning_instruction, thought_traj, action_traj)

                            # DOUBLE CHECK if rephrasing thought also change the original action
                            keep_traj = align_thought_action(action_traj, new_thought_traj)
                            if not keep_traj:
                                print("action doesn't match thought")
                                return None

                            new_traj = replace_thought(trajectory, new_thought_traj)
                            analysis = analysis_thought(terminal_state, step_history)

                            new_traj[-1] = (terminal_state, f"Let's think step by step. {analysis} I think I'm ready to answer the question: {post_reasoning_instruction}. {explanation}", f"stop [{traj_answer}]", "stop", new_traj[-1][4])
                            thought_action_traj[-1]['Action'] = f"stop [{traj_answer}]"

                            directory = f'train_set_real/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
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
    return high_level_intent


############################################
###### Real Env Collection ######
############################################
def ensure_login(domain: str):
    """Ensure login before every call to webarena_real_traj."""
    subprocess.run(["python", "webarena/browser_env/auto_login.py", "--site_list", domain], check=True)

def collect_real_env(domain: str,
                     file_idx: int,
                     min_step: int,
                     index: int,
                     num: int,
                     num_guides_per_step):
    env, init_state = prepare_webarena_env(file_idx)
    # propose the guide
    state = init_state['observation']['text'].split('\n\n')[1]
    for i in range(num):
        if domain != "map":
            ensure_login(domain)
        tasks = task_guidance(state, num_guides=num_guides_per_step, domain=domain)
        guide = random.choice(tasks)
        webarena_real_traj(env, init_state, domain, guide, num_guides_per_step, min_step=min_step, max_step=min_step+6, index=index, num=i, save=True, general_searching=0.5)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default='shopping')
    parser.add_argument('--file_index', type=int)
    parser.add_argument('--min_step', type=int, default=1)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--num_guides_per_step', type=int, default=5)

    args = parser.parse_args()
    domain = args.domain
    file_idx = args.file_index
    min_step = args.min_step
    index = args.index
    num = args.num
    
    num_guides_per_step = args.num_guides_per_step

    print(f"Collecting real envs over domain {domain} with file index {file_idx}, min step {min_step}, index {index}, num {num}")
    # real env collection
    collect_real_env(domain, file_idx, min_step, index, num, num_guides_per_step)