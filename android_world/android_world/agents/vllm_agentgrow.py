# Copyright 2024 The android_world Authors.
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

"""T3A: Text-only Autonomous Agent for Android."""

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

import re
import requests
import json
import random

SYS_PROMPT = """
Assume you are browsing on an Android mobile phone. You want to acheive the "Goal" given below.
Given the goal, the current page and the browsing history, your task is to think about how to acheive the goal, continue the browsing by giving an action in current timestep, and generate a step-wise abstract to conclude what this action does.

Available actions are: 
click [id]: click on element with id on current page.
input_text [id] [content]: use keyboard to write "content" into a text box with id.
open_app [app_name]: open app with name app_name.
keyborad_enter: press enter key
scroll[direction]: move current Web in a specific direction. You can choose from scroll('up'), scroll('down'), scroll('left'), scroll('right').
navigate_back: go back to last Web page
navigate_home: go to the root Web page
wait: do nothing during a specific time length
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide empty answer in the bracket.
"""


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





class AgentgrowVllm(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      model_name: str = 'meta-llama/Meta-Llama-3-8B',
      name: str = 'AgentgrowVllm',
      remove_metadata: bool = False,
      do_shuffle: bool = False,
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    # load the llm via vllm 
    self.model_name = model_name
    self.history = []
    self.browsing_history = []
    self.additional_guidelines = None
    self.remove_metadata = remove_metadata
    self.stopped = False
    self.do_shuffle = do_shuffle


  def call_llm(self, prompt: str) -> str:

    server_url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}


    data = {
        "model": self.model_name,
        "messages": [
           {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.6,
    }

    try:
      response = requests.post(server_url, headers=headers, json=data)
      result = response.json()
      print("Response:", result)
      return (
        result["choices"][0]["message"]["content"],
        True,
        result
      )
    except:
      print("Error in response:", result)
      return None, False, None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []
    self.browsing_history = []
    self.stopped = False

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def _parse_action(self, action: str, index_map: dict = None) -> dict:
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

  def _parse_output(self, output:str):
    thought, action, summary = output.strip().split('\n')
    thought = thought.split(':')[1].strip()
    action = action.split(':')[1].strip()
    summary = summary.split(':')[1].strip()

    action = self._parse_action(action)

    return thought, action, summary
    
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size

    ui_elements = state.ui_elements
    before_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements


    index_map = None
    if self.remove_metadata:
      new_state = ''
      for line in before_element_list.strip().split('\n'):
        
        idx = line.index(', class_name')
        line = line[:idx] + ')'
        new_state += line + '\n'
    elif self.do_shuffle:
      index_map = {}
      element_list = before_element_list.strip().split('\n')
      random.shuffle(element_list)
      new_state = ''
      for i, line in enumerate(element_list):
        orig_index = int(line.split(':')[0].split('UI element ')[1])
        index_map[i] = orig_index
        new_state += f'UI element {i}: ' + ': '.join(line.split(': ')[1:]) + '\n'
    else:
      new_state = before_element_list
    if not self.stopped:
      action_prompt = """Current page:
{}

Goal: {}

Browsing history:
{}""".format(new_state, goal, '\n'.join(self.browsing_history))

      step_data['action_prompt'] = action_prompt

      output, is_safe, raw_response = self.call_llm(
          action_prompt,
      )

      # parse the output to get the action, the thought and the step summary.
      thought, action, summary = self._parse_output(output, index_map)
      if action['action_type'] == 'answer' or action['action_type'] == 'status':
        self.stopped = True
      action = str(action)
      print('Thought: ' + thought)
      print('Action: ' + action)
      print('Summary: ' + summary)

      self.browsing_history.append(summary)

      
    else:
       action = '{"action_type": "status", "goal_status": "complete"}'
       thought = "The task is complete."
       summary = "The task is complete."
       output = "The task is complete."
       raw_response = "The task is complete."

    step_data['action_output'] = output
    step_data['parsed_action'] = action
    step_data['thought'] = thought
    step_data['action_raw_response'] = raw_response

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if (not thought) or (not action):
      print('Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    if converted_action.action_type in ['click', 'long-press', 'input-text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        # Add mark for the target ui element, just used for visualization.
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
        )

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    after_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        self.env.logical_screen_size,
    )

    # Save screenshot only for result visualization.

    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements

    step_data['summary'] = (
        f'Summary: {summary} '
        if raw_response
        else 'Error calling LLM in summerization phase.'
    )
    print('Summary: ' + summary)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
