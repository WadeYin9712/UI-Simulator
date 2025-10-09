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

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import openai
import hashlib
from openai import OpenAI
import os
import copy
import re
import requests
import json

SYS_PROMPT = """
Assume you are using a Android mobile phone
Based on the current UI page, and the action that has been taken in other tasks, your task is to analyze the current page, and continue browsing that is different from previous steps in other tasks by giving the action on current page.
Here are some requirments for output:
    - The action should strictly follow the action format given action space.
    - You also need to generate a single sentence abstract to summarize what this action does.
    - Thought, Action and Task should not exceed one line.

Available actions:
click [id]: click on element with id on current page.
input_text [id] [content]: use keyboard to write "content" into a text box with id.
open_app [app_name]: open app with name app_name.
keyborad_enter: press enter key
scroll [direction]: move current Web in a specific direction. You can choose from scroll [up], scroll [down], scroll [left], scroll [right].
navigate_back: go back to last Web page
navigate_home: go to the root Web page
wait: do nothing during a specific time length
stop [answer]: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide empty answer in the bracket.

Example Input:
Guide: Create a new contact for "James Brown".
Current state:
Element 1: UIElement(text=None, content_description=None, class_name=android.view.View, bbox=None, bbox_pixels=BoundingBox(x_min=0, x_max=1080, y_min=0, y_max=2400), hint_text=None, is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True, is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/background_container, tooltip=None, resource_id=None, metadata=None)
Element 2: UIElement(text=First Name, content_description=James, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=100, y_max=150), hint_text=Enter first name, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/first_name, tooltip=None, resource_id=None, metadata=None)
Element 3: UIElement(text=Last Name, content_description=Brown, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=160, y_max=210), hint_text=Enter last name, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/last_name, tooltip=None, resource_id=None, metadata=None)
Element 4: UIElement(text=Phone Number, content_description=None, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=220, y_max=270), hint_text=Enter phone number, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/phone_number, tooltip=None, resource_id=None, metadata=None)
Element 5: UIElement(text=Email Address, content_description=None, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=280, y_max=330), hint_text=Enter email address, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/email_address, tooltip=None, resource_id=None, metadata=None)
Element 6: UIElement(text=Save, content_description=None, class_name=android.widget.Button, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=520, y_min=360, y_max=410), hint_text=None, is_checked=False, is_checkable=False, is_clickable=True, is_editable=False, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/save_button, tooltip=None, resource_id=None, metadata=None)
Element 7: UIElement(text=Cancel, content_description=None, class_name=android.widget.Button, bbox=None, bbox_pixels=BoundingBox(x_min=560, x_max=1040, y_min=360, y_max=410), hint_text=None, is_checked=False, is_checkable=False, is_clickable=True, is_editable=False, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/cancel_button, tooltip=None, resource_id=None, metadata=None)
Element 8: UIElement(text=Add to Favorites, content_description=None, class_name=android.widget.Switch, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=300, y_min=430, y_max=480), hint_text=None, is_checked=False, is_checkable=True, is_clickable=True, is_editable=False, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/add_to_favorites_switch, tooltip=None, resource_id=None, metadata=None)
Element 9: UIElement(text=Additional Information, content_description=None, class_name=android.widget.TextView, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=490, y_max=540), hint_text=None, is_checked=False, is_checkable=False, is_clickable=True, is_editable=False, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/additional_info_header, tooltip=None, resource_id=None, metadata=None)
Element 10: UIElement(text=Address, content_description=None, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=550, y_max=600), hint_text=Enter address, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/address, tooltip=None, resource_id=None, metadata=None)
Element 11: UIElement(text=Notes, content_description=None, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=610, y_max=660), hint_text=Enter notes, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/notes, tooltip=None, resource_id=None, metadata=None)
Element 12: UIElement(text=Company, content_description=None, class_name=android.widget.EditText, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=1040, y_min=670, y_max=720), hint_text=Enter company name, is_checked=False, is_checkable=False, is_clickable=True, is_editable=True, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/company, tooltip=None, resource_id=None, metadata=None)
Element 13: UIElement(text=None, content_description=Profile picture placeholder, class_name=android.widget.ImageView, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=150, y_min=750, y_max=850), hint_text=None, is_checked=False, is_checkable=False, is_clickable=True, is_editable=False, is_enabled=True, is_focused=False, is_focusable=True, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/profile_picture, tooltip=None, resource_id=None, metadata=None)
Element 14: UIElement(text=Format: example@domain.com, content_description=None, class_name=android.widget.TextView, bbox=None, bbox_pixels=BoundingBox(x_min=40, x_max=300, y_min=790, y_max=830), hint_text=None, is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True, is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.google.android.contacts, resource_name=com.google.android.contacts:id/email_format_tooltip, tooltip=None, resource_id=None, metadata=None)
Element 15: UIElement(text=15:34, content_description=Current time, class_name=android.widget.TextView, bbox=None, bbox_pixels=BoundingBox(x_min=900, x_max=1080, y_min=10, y_max=50), hint_text=None, is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True, is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=False, is_selected=False, is_visible=True, package_name=com.android.systemui, resource_name=com.android.systemui:id/clock, tooltip=None, resource_id=None, metadata=None)

Previously taken actions in other tasks:
Open the Contacts app.
Click "Create contact" to initiate the process of creating a new contact
Type "James" to the First Name
Type "Brown" to the Last Name

Example Output:
Thought: Let's think step by step. The guide is 'Create a new contact for "James Brown". From previous steps, I opened the 'Contacts' app, started the creation process and typed the first and last name. The current page shows that I've successfully typed the First and last name, and I also need to fill in details like phone number, email address. Since the guide doesn't provide the phone number, I should give a realistic phone number here, like "718-099-5256". To continue creating the contact, I shall type "718-099-5256" to the Phone number.
Action: input_text [4][718-099-5256]
Task: Type "718-099-5256" as the phone number.
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





class AgentgrowExplore(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      model_name: str = 'gpt-4o-mini',
      name: str = 'AgentgrowVllm',
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
    self.stopped = False
    self.step_records = []
    self._just_opened_app = False
    self._current_app_name = None
    self._current_merged_elements = []
    self._executed_actions = set()
    self._step_count = 0
    self._known_state_actions = {}  # {state_hash: set([action_str])}

  def call_llm(self, prompt, model='gpt-4o-mini', stop=None, return_json=False, max_tokens=None, temperature=0.5):
    client = OpenAI(
        organization=os.environ['OPENAI_ORG_ID'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" } if return_json else openai.NOT_GIVEN,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature
    )
    return completion.choices[0].message.content, True, completion
  
  def load_known_actions_from_trajectories(self, state_dir_root="states"):
    """Scan previous step_*.json files and load state->action history."""
    self._known_state_actions = {}

    for app_dir in os.listdir(state_dir_root):
        app_path = os.path.join(state_dir_root, app_dir)
        if not os.path.isdir(app_path):
            continue
        for fname in os.listdir(app_path):
            if not fname.startswith("step_") or not fname.endswith(".json"):
                continue
            fpath = os.path.join(app_path, fname)
            with open(fpath, "r") as f:
                step_data = json.load(f)
            elements = step_data.get("prev_elements")
            actions = step_data.get("action_history", [])
            action = actions[-1] if actions else None
            if not elements or not actions:
                continue
            try:
                state_hash = hashlib.sha256(elements[0].encode()).hexdigest()
                if state_hash not in self._known_state_actions:
                    self._known_state_actions[state_hash] = set()
                self._known_state_actions[state_hash].update(action)
            except Exception as e:
                print(f"Failed to load from {fpath}: {e}")

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []
    self.browsing_history = []
    self._step_count = 0
    self._current_app_name = None
    
    self.stopped = False
    self.load_known_actions_from_trajectories(state_dir_root="intermediate_states")

    state = self.get_post_transition_state()
    ui_elements = state.ui_elements
    self._current_merged_elements = ui_elements
    

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines
  

  def _parse_action(self, action: str) -> dict:
    # parse the action to the expected format
    action = action.strip()

    if action.startswith("click"):
        match = re.match(r"click\s*\[(.*?)\]", action)
        if match:
            target_index = int(match.group(1))
            return {"action_type": "click", "index": target_index}

    elif action.startswith("input_text"):
        match = re.match(r"input_text\s*\[(.*?)\]\s*\[(.*?)\]", action)
        if match:
            target_index, text_input = int(match.group(1)), match.group(2)
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
    
  def _fingerprint_ui_state(self, elements: list) -> str:
    """Generate a fingerprint of the merged UI state."""
    simplified = [
        (el.resource_name, el.text, el.class_name)
        for el in elements
    ]
    state_str = json.dumps(simplified, sort_keys=True)
    return hashlib.sha256(state_str.encode()).hexdigest()

  def _parse_output(self, output:str):
    thought, action, summary = [s for s in output.strip().split('\n') if s]
    thought = thought.split(':')[1].strip()
    action = action.split(':')[1].strip()
    summary = summary.split(':')[1].strip()

    action = self._parse_action(action)

    return thought, action, summary
  
  def _merge_scrolled_ui_elements(self, states: list[list], scroll_height: int = 2400) -> list:
    """Merge UI elements across multiple scroll steps. Adjust Y coords for new elements."""
    base, scroll1, scroll2 = states
    merged = list(base)
    seen = set()

    def _element_key(el):
        return (el.resource_name, el.text, el.class_name)

    def _bbox_shift(bbox, y_offset):
        if bbox:
            return representation_utils.BoundingBox(
                x_min=bbox.x_min,
                x_max=bbox.x_max,
                y_min=bbox.y_min + y_offset,
                y_max=bbox.y_max + y_offset,
            )
        return None

    for i, state in enumerate([scroll1, scroll2], start=1):
        for el in state:
            key = _element_key(el)
            if any(_element_key(e) == key for e in merged):
                continue  # skip if already in merged (persistent UI element)

            new_el = copy.deepcopy(el)
            if new_el.bbox_pixels:
                new_el.bbox_pixels = _bbox_shift(new_el.bbox_pixels, i * scroll_height)
            merged.append(new_el)

    return merged

    
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    if self._step_count >= 6:
      print('Agent has reached the maximum number of steps.')
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
      step_data['summary'] = "The task is infeasible."
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)
    original_state = self.get_post_transition_state()
    original_elements = original_state.ui_elements
    
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
    if self._just_opened_app:
        before_element_list = _generate_ui_elements_description_list_full(
            self._current_merged_elements,
            logical_screen_size,
        )
    else:
        before_element_list = _generate_ui_elements_description_list_full(
            original_elements,
            logical_screen_size,
    )

    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements

    state_hash = self._fingerprint_ui_state(self._current_merged_elements)
    previous_actions = self._known_state_actions.get(state_hash, set())

    if not self.stopped:
      if self._current_app_name:
        action_prompt = """Current page:
{}

Browsing history: {}

Previously taken actions in other tasks: {}\nNote: Don't try to make phone call to others!""".format(before_element_list, '\n'.join(self.browsing_history), '\n'.join(list(previous_actions)))
        step_data['action_prompt'] = action_prompt
      else:
        action_prompt = """Current page:
{}

Reference Goal: {}\nNote: You should open the app as indicated in the reference goal by only taking the open_app action. Don't try to do other actions.
Available app names:
Settings
Clock
Audio Recorder
Broccoli
Simple Calendar Pro
Camera
Contacts
Simple Draw Pro
Files
Simple Gallery Pro
Joplin
Markor
OpenTracks
OsmAnd
Pro Expense
Retro Music
Simple SMS Messenger
Tasks
VLC
""".format(before_element_list, goal)
      
      output, is_safe, raw_response = self.call_llm(
          action_prompt,
      )

      # parse the output to get the action, the thought and the step summary.
      thought, action, summary = self._parse_output(output)
      print(action)
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
      idx = converted_action.index
      screen_height = self.env.logical_screen_size[1]
      try:
          target_el = self._current_merged_elements[idx]
      except IndexError:
          print(f"Invalid index {idx} in merged elements.")
          step_data['summary'] = "Invalid index."
          self.history.append(step_data)
          return base_agent.AgentInteractionResult(False, step_data)

      # If y_min > screen height, element is off screen
      if target_el.bbox_pixels and target_el.bbox_pixels.y_min > screen_height:
          print("Element is off-screen, will scroll into view.")

          # Scroll until found (e.g. assume 2 max tries)
          found = False
          for _ in range(2):
              self.env.execute_action(json_action.JSONAction(action_type="scroll", direction="down"))
              env_elements = self.get_post_transition_state().ui_elements
              for new_idx, el in enumerate(env_elements):
                  if (
                      el.resource_name == target_el.resource_name and
                      el.text == target_el.text and
                      el.class_name == target_el.class_name
                  ):
                      converted_action.index = new_idx
                      found = True
                      break
              if found:
                  break

          if not found:
              print("Unable to locate element on screen after scrolling.")
              step_data['summary'] = "Could not find target element after scroll."
              self.history.append(step_data)
              return base_agent.AgentInteractionResult(False, step_data)

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

    #   if converted_action.action_type == "open_app":
    #     self._just_opened_app = True
    #     self._current_app_name = converted_action.app_name

    #     base_state = self.get_post_transition_state()
    #     base_elements = base_state.ui_elements

    #     scrolled_states = [base_elements]

    #     for _ in range(2):
    #       self.env.execute_action(json_action.JSONAction(action_type="scroll", direction="down"))
    #       scrolled_state = self.get_post_transition_state()
    #       scrolled_states.append(scrolled_state.ui_elements)

    #     merged_elements = self._merge_scrolled_ui_elements(scrolled_states)
    #     self._current_merged_elements = merged_elements

    #     for _ in range(2):
    #       self.env.execute_action(json_action.JSONAction(action_type="scroll", direction="up"))

    #     ui_elements = merged_elements

    #   else:
      self._just_opened_app = False
      state = self.get_post_transition_state()
      ui_elements = state.ui_elements
      self._current_merged_elements = ui_elements

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

    step_json = {
        "prev_elements": [_generate_ui_elements_description_list_full(original_elements, logical_screen_size)],
        "action_history": list(self.browsing_history),
        "next_elements": [_generate_ui_elements_description_list_full(ui_elements, logical_screen_size)],
    }

    dir = f"intermediate_states/{self._current_app_name}"
    os.makedirs(dir, exist_ok=True)
    # index of the step: number of the steps in the app directory
    idx = len(os.listdir(dir))
    with open(f"intermediate_states/{self._current_app_name}/step_{idx}.json", "w") as f:
        json.dump(step_json, f, indent=4)
    self.step_records.append(step_json)
    self._step_count += 1

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
