"""
Code for Ablation 1: w/o step-wise task control

"""
from actions.action import *
from simulator import *
import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import re
from android_collector import *
from utils import *
import traceback
import argparse

@timer
def android_sim_traj_ablation1(init_state: str,
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
    early_stop_flag = False
    early_stop_thought = ""
    completion = False
    general_flag = False
        
    guide = "None"
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
        directory = f'train_set_android_ablation1/init_state_{index}_maxlen={max_step}/iter_{num}'
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

                            directory = f'train_set_android_ablation1/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
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
###### Data collection functions ######
############################################
def process_sim_collect_ablation1(init_state, num, min_step=3, max_step=5, index=1, save=True, rag_enabled=False):
    # print(guides)
    
    if num == 0:
        return None
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(
            android_sim_traj_ablation1,
            [init_state] * num,
            [min_step] * num,
            [max_step] * num,
            [index] * num,
            [i for i in range(num)],
            [save] * num,
            [rag_enabled] * num
        )
    return results
        
def collect_ablation1(init_states, min_length_range: list, nums: list, debug=False, rag_enabled=True):

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

        for i in range(nums[0]):
            android_sim_traj_ablation1(init_states[0], min_step=min_step, max_step=min_step+6, index=1000, num=i, save=True, rag_enabled=rag_enabled)

    else:
        # Propose first-step guides
        filtered_init_states = []
        for state in init_states:
            simulator = AndroidSimulator()
            try:
                simulator.reset(state)
            except Exception as e:
                print("Unexpected Error:", e)
                continue

            filtered_init_states.append(state)

        # collect first-round trajectories
        for l, num in zip(min_length_range, nums):
            
            with ProcessPoolExecutor(max_workers=5) as executor:
                results = executor.map(
                    process_sim_collect_ablation1,
                    filtered_init_states,
                    [num] * len(filtered_init_states),
                    [l] * len(filtered_init_states),
                    [l+6] * len(filtered_init_states),
                    [i for i in range(len(filtered_init_states))],
                    [True] * len(filtered_init_states),
                    [rag_enabled] * len(filtered_init_states)
                )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rag_enabled', action='store_true')
    parser.add_argument('--min_steps', type=int, default=1, nargs='+')
    parser.add_argument('--nums', type=int, default=10, nargs='+')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if isinstance(args.min_steps, int):
        args.min_steps = [args.min_steps]
    if isinstance(args.nums, int):
        args.nums = [args.nums]
    init_states = get_init_states()

    collect_ablation1(init_states, args.min_steps, args.nums, debug=args.debug, rag_enabled=args.rag_enabled)
