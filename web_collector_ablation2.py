"""
Code for Ablation 2: w/o multi-step simulation

"""
import random
import actions
from actions.action import *
from simulator import *
import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from converter import WebArenaConverter
from web_collector import *
from utils import *
import traceback
import argparse

@timer
def webarena_sim_traj_ablation2(init_state: dict, 
                      domain: str,
                      guide, 
                      num_guides_per_step=5,
                      min_step=3, 
                      max_step=5, 
                      index=1, 
                      num=1, 
                      save=True,
                      webarena_obs_format=True,
                      rag_enabled=False,
                      general_entity=0.5):
    if rag_enabled:
        print("####RAG enabled####")
        simulator = RAG_Simulator(webarena_domain=domain, webarena_mode=True)
    else:
        print("####RAG disabled####")
        simulator = AblationSimulator(webarena_mode=True)
    simulator.reset(init_state)
    converter = WebArenaConverter()
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
        
    # check if guide is search related
    result = judge_missed_entity(guide)
    
    if result:
        a11y_tree = converter.convert_tree_venv_to_real_env(init_state)
        if random.random() < general_entity:
            guide = general_entity(a11y_tree, guide)
            general_flag = True
        else:
            guide = specify_entity(a11y_tree, guide)
    
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
        
        # get the whole state
        state = simulator.remove_coord_info(simulator.flatten_tree_state(simulator._get_backup_current_state()))
        print(f"############# index {index}, iter {num}, step {i} #############")
        error = False
        early_stop = i > min_step
        # generate thought and action at current step
        print("guide for thought action generation:", guide)
        try:
            thought, action, task = thought_action_gen(domain, state, guide, step_history, early_stop)
        except:
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
            action = match_webarena_action(action)  # class
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("original action:", action)
            action = reformat_action(action, actions.AVAILABLE_WEBARENA_ACTIONS)
            action = match_webarena_action(action)

        # post process the action
        cur_actions = []
        if (isinstance(action, WA_Click) or isinstance(action, WA_Type) or isinstance(action, WA_Hover)):
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

            try:
                coord = simulator.get_element_coordinates(target_element)
            except:
                print(traceback.format_exc())
                error = True
                retry += 1
                continue

            if 'up' in coord and (coord['up'] + simulator.coord_offset[simulator.tab_index][1]) >= 1080:
                scroll_time = int((coord['up'] + simulator.coord_offset[simulator.tab_index][1] - 1080) // 1080 + 1)
                cur_actions += [WA_Scroll('down')] * scroll_time
                print("scroll down", scroll_time)
            elif 'down' in coord and (coord['down'] + simulator.coord_offset[simulator.tab_index][1]) < 0:
                scroll_time = int(abs((coord['down'] + simulator.coord_offset[simulator.tab_index][1]) // 1080) + 1)
                cur_actions += [WA_Scroll('up')] * scroll_time
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

        prev_a11y_tree_state = converter.convert_tree_venv_to_real_env(simulator.erase_coord_info_in_tree(simulator._get_backup_current_state(), duplicate=True))
        # execute the actions
        for action in cur_actions:
            prev_state = copy.deepcopy(simulator.cur_state)
            new_state = None

            try:
                if webarena_obs_format:
                    # convert back to the original format
                    depths = simulator._get_depths(simulator._get_backup_current_state())
                    WA_prev_state = converter.convert_flat_tree_venv_to_real_env(prev_state, depths)
                if isinstance(action, WA_Scroll):
                    analysis = analysis_thought(simulator.cur_state, step_history)
                    step_thought = "Let's think step by step. " + analysis + f"I think the content I want is not appearing in current window, but it should be on the current webpage. So I'll scroll down to find more information. In summary, the next action I will perform is scroll [{action.direction}]"
                    step_task = f"Scroll {action.direction} the current page to find more information on the page."
                else:
                    step_thought = thought
                    step_task = task

                # RAG: simulate next state with action history
                if rag_enabled:
                    if not isinstance(action, WA_Scroll):
                        try:
                            element = find_element_cleaned(WA_prev_state, action.id)
                        except:
                            element = None
                        cur_action = f"{str(action).split(' ')[0]} {element}"
                        if isinstance(action, WA_Type) and action.press_enter_after == 1:
                            cur_action += " press enter"
                        new_state = simulator.step(action, action_history + [cur_action])
                        action_history.append(cur_action)
                    else:
                        new_state = simulator.step(action, '')

                else:
                    new_state = simulator.step(action)

                a11y_tree_state = converter.convert_tree_venv_to_real_env(simulator.erase_coord_info_in_tree(simulator._get_backup_current_state(), duplicate=True))
                print("*************")
                print(a11y_tree_state)
                print("*************")
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
            webarena_action = str(action)
            thought_action = {
                f"Thought": step_thought,
                f"Action": str(action)
            }

            history = copy.deepcopy(step_history)
            tmp_traj = (WA_prev_state, step_thought, webarena_action, step_task, history)
            thought_action_traj.append(thought_action)
            step_history.append(step_task)
            
            # print(i, history)
            trajectory.append(tmp_traj)

            if not isinstance(action, WA_Scroll):
                # check if general guide is completed
                cur_state = converter.convert_tree_venv_to_real_env(simulator.erase_coord_info_in_tree(simulator._get_backup_current_state(), duplicate=True))
                completion = judge_guide_completion(guide, step_history, prev_a11y_tree_state, cur_state)
                print("check guide completion:", completion)
                
                # generate task guide if guide is completed
                if allowed_modify_time > 0 and completion and i <= 2 * min_step:
                    # modify the guide
                    a11y_tree_state = converter.convert_tree_venv_to_real_env(simulator.erase_coord_info_in_tree(simulator._get_backup_current_state(), duplicate=True))

                    # generate task guide if guide is completed
                    if completion:
                        tmp = task_guidance(a11y_tree_state, prev_a11y_tree_state, guide, step_history, first_guide=False, num_guides=num_guides_per_step)
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
        analysis = analysis_thought(simulator.cur_state, step_history)
        # print("###### analysis ######")
        # print(analysis)
        # print("############")
        stop_thought = f"Let's think step by step. {analysis} I think I've completed the task. The action I'll take is stop []."

    stop_action = 'stop []' if 'unachievable' not in webarena_action else 'stop [unachievable]'

    terminal_state = simulator.cur_state
    if webarena_obs_format:
        depths = simulator._get_depths(simulator._get_backup_current_state())
        terminal_state = converter.convert_flat_tree_venv_to_real_env(terminal_state, depths)
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


        if rag_enabled:
            directory = f'train_set_{domain}_ablation2_rag/init_state_{index}_maxlen={max_step}/iter_{num}'
        else:
            directory = f'train_set_{domain}_ablation2/init_state_{index}_maxlen={max_step}/iter_{num}'
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
                terminal_state = simulator._get_backup_current_state()
                if webarena_obs_format:
                    terminal_state = converter.convert_tree_venv_to_real_env(terminal_state)
                
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
                                continue

                            new_traj = replace_thought(trajectory, new_thought_traj)
                            analysis = analysis_thought(simulator.get_visible_elements(simulator._get_backup_current_state()), step_history)

                            new_traj[-1] = (terminal_state, f"Let's think step by step. {analysis} I think I'm ready to answer the question: {post_reasoning_instruction}. {explanation}", f"stop [{traj_answer}]", "stop", new_traj[-1][4])
                            thought_action_traj[-1]['Action'] = f"stop [{traj_answer}]"

                            if rag_enabled:
                                directory = f'train_set_{domain}_ablation2_rag/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
                            else:
                                directory = f'train_set_{domain}_ablation2/init_state_{index}_maxlen={max_step}/iter_{num}_reasoning_{i}'
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
def process_sim_collect_ablation2(init_state, domain, guides, num_guides_per_step, min_step=3, max_step=5, index=1, save=True, rag_enabled=False):
    # print(guides)
    
    num = len(guides)
    if num == 0:
        return None
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(
            webarena_sim_traj_ablation2,
            [init_state] * num,
            [domain] * num,
            guides,
            [num_guides_per_step] * num,
            [min_step] * num,
            [max_step] * num,
            [index] * num,
            [i for i in range(num)],
            [save] * num,
            [True] * num,
            [rag_enabled] * num
        )

    return results
        
def collect_ablation2(domain, init_states, num_guides_per_step, min_length_range: list, nums: list, debug=False, rag_enabled=True):

    if debug:
        min_step = min_length_range[0]
        for state in init_states[:1]:
            simulator = Simulator(webarena_mode=True)
            try:
                simulator.reset(state)
            except Exception as e:
                print("Unexpected Error:", e)
                init_states.remove(state)
                continue

            # convert the state into a11y tree
            converter = WebArenaConverter()
            a11y_tree = converter.convert_tree_venv_to_real_env(state)
            tasks = task_guidance(a11y_tree, num_guides = num_guides_per_step, domain=domain)
            print(tasks)

        guide = random.sample(tasks, 1)[0]
        
        for i in range(nums[0]):
            webarena_sim_traj_ablation2(init_states[0], domain, guide, num_guides_per_step, min_step=min_step, max_step=min_step+6, index=1000, num=i, save=True, rag_enabled=rag_enabled)

    else:
        # Propose first-step guides
        guides = []
        filtered_init_states = []
        for state in init_states:
            simulator = Simulator(webarena_mode=True)
            try:
                simulator.reset(state)
            except Exception as e:
                print("Unexpected Error:", e)
                continue

            # convert the state into a11y tree
            converter = WebArenaConverter()
            a11y_tree = converter.convert_tree_venv_to_real_env(state)
            tasks = task_guidance(a11y_tree, num_guides = num_guides_per_step, domain=domain)
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
                    process_sim_collect_ablation2,
                    filtered_init_states,
                    [domain] * len(filtered_init_states),
                    sampled_guides,
                    [num_guides_per_step] * len(filtered_init_states),
                    [l] * len(filtered_init_states),
                    [l+6] * len(filtered_init_states),
                    [i for i in range(len(filtered_init_states))],
                    [True] * len(filtered_init_states),
                    [rag_enabled] * len(filtered_init_states)
                )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='general', choices=['shopping', 'gitlab', 'reddit', 'map', 'shopping_admin', 'general'])
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
    init_states = get_init_states(args.domain)
    # don't do random shuffling!

    collect_ablation2(args.domain, init_states, args.num_guides_per_step, args.min_steps, args.nums, debug=args.debug, rag_enabled=args.rag_enabled)