"""
Code for simulator implementation, including web simulator and android simulator.
"""
import json
from actions.action import *
from actions.webarena_action import *
import os
from converter import *
from openai import OpenAI
import openai
import copy
import random
import logging
from utils import *
import re
from typing import Optional

STATIC_ELEMENT = {
    "text",
    "div",
    "main",
    "RootWebArea"
}

####################################
########## Web Simulator ##########
####################################

class Simulator:
    def __init__(self,
                 init_state=None,
                 width=1920,
                 height=1080,
                 webarena_mode=False,
                 debug=False,
                 ):
        self.client = OpenAI(
            organization=os.environ['OPENAI_ORG_ID'],
            api_key=os.environ['OPENAI_API_KEY']
        )

        self.init_state = init_state

        self.cur_state = None
        self.home_state = None

        # use array of arrays to store previous states
        self.prev_states = []
        self.tab_index = -1
        self.window_index = []
        # offset for each tab
        self.coord_offset = []
        self.steps = 0

        # simulation history
        self.intent_history = []
        self.key_infos = []

        # screen parameters
        self.width = width
        self.height = height
        self.style_hint = ''

        self.webarena_mode = webarena_mode
        self.webarena_converter = WebArenaConverter()

        # logging module
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.INFO,
                                filemode='w',
                                filename='tmp/simulator.log',
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

    def call_openai(self,
                    prompt, 
                    sys_prompt, 
                    model="gpt-4o-mini",
                    stop=None, 
                    return_json=False,
                    max_tokens=None,
                    temperature=0.5):
        completion = self.client.chat.completions.create(
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
        caller = inspect.stack()[1].function
        usage = completion.usage
        # print(f"Function ##{caller}## called with usage: Input token:{usage.prompt_tokens}, Output token:{usage.completion_tokens}")

        return completion.choices[0].message.content
    
    
    # Function to locate an element by coordinates
    @staticmethod
    def find_element_by_coordinates(elements, x, y, filter_static=False, filter_image=False):
        if isinstance(elements, dict):
            elements = [elements]
        for element in elements:
            coord = element['coord']
            if coord['left'] <= x <= coord['right'] and coord['up'] <= y <= coord['down']:
                if not (filter_static and element['tag'] in STATIC_ELEMENT) and not (filter_image and element['tag'] == 'img'):
                    return element
                else:
                    return None
                
            elif 'elements' in element:
                e = Simulator.find_element_by_coordinates(element['elements'], x, y, filter_static, filter_image)
                if e:
                    return e
                
        return None
    @staticmethod
    def find_element_by_id(elements, id, filter_static=False, filter_image=False):
        if isinstance(elements, dict):
            elements = [elements]
        for element in elements:
            if element['coord']['id'] == id:
                if not ((filter_static and element['tag'] in STATIC_ELEMENT) or (filter_image and element['tag'] == 'img')):
                    return element
                else:
                    return None
            elif 'elements' in element:
                e = Simulator.find_element_by_id(element['elements'], id, filter_static, filter_image)
                if e:
                    return e
        return None
    
    def _get_leaf_node(self, elements):
        # get one leaf node from the tree
        if isinstance(elements, dict):
            elements = [elements]
        for element in elements:
            if 'elements' in element:
                return self._get_leaf_node(element['elements'])
            else:
                return element
            
    def _get_depths(self, state, depth=0) -> dict:
        # first-order traverse the tree to get the depth of each node
        if isinstance(state, dict):
            state = [state]

        depths = {}
        for element in state:
            if 'elements' in element:
                child_depths = self._get_depths(element['elements'], depth + 1)
                for key, value in child_depths.items():
                    depths[key] = value
            depths[element['coord']['id']] = depth
        return depths

            
    def shuffle_element_id(self, elements):
        if isinstance(elements, dict):
            elements = [elements]
        for element in elements:
            if 'coord' in element:
                element['coord']['id'] = random.randint(0, 20000)
            if 'elements' in element:
                self.shuffle_element_id(element['elements'])
        return elements[0]

    def check_element_visibility(self, element=None, coord=None):
        
        if not coord:
            coord = copy.deepcopy(element['coord'])
        offset = self.coord_offset[self.tab_index]
        coord['left'] += offset[0]
        coord['right'] += offset[0]
        coord['up'] += offset[1]
        coord['down'] += offset[1]

        # check if the element is visible
        return coord['left'] < self.width and coord['right'] > 0 and coord['up'] < self.height and coord['down'] > 0

    def get_element_coordinates(self, element) -> dict:
        if 'coord' in element and 'left' in element['coord']:
            return {
                'left': element['coord']['left'],
                'right': element['coord']['right'],
                'up': element['coord']['up'],
                'down': element['coord']['down']
            }
        elif 'elements' in element:
            # aggregate the coordinates of the child nodes
            coord = {
                'left': 19200,
                'right': -10,
                'up': 10800,
                'down': -10
            }
            for e in element['elements']:
                child_coord = self.get_element_coordinates(e)
                coord['left'] = min(coord['left'], child_coord['left'])
                coord['right'] = max(coord['right'], child_coord['right'])
                coord['up'] = min(coord['up'], child_coord['up'])
                coord['down'] = max(coord['down'], child_coord['down'])
            return coord
    
        raise ValueError("Element does not have coordinates.")

    def get_visible_elements(self, elements) -> list:

        # Return: a list of visible elements, the tree structure is flattened.

        visible_elements = []
        if isinstance(elements, dict):
            elements = [elements]
        for element in elements:
            if 'elements' in element:
                coord = self.get_element_coordinates(element)
                if self.check_element_visibility(coord=coord):
                    cur_element = copy.deepcopy(element)
                    del cur_element['elements']
                    visible_elements.append(cur_element)
                    visible_elements.extend(self.get_visible_elements(element['elements']))
                else:
                    continue
            
            elif self.check_element_visibility(element=element):
                visible_element = copy.deepcopy(element)
                # if up > down or left > right, reverse them
                if visible_element['coord']['up'] > visible_element['coord']['down']:
                    visible_element['coord']['up'], visible_element['coord']['down'] = visible_element['coord']['down'], visible_element['coord']['up']
                if visible_element['coord']['left'] > visible_element['coord']['right']:
                    visible_element['coord']['left'], visible_element['coord']['right'] = visible_element['coord']['right'], visible_element['coord']['left']
                
                coord = copy.deepcopy(element['coord'])
                offset = self.coord_offset[self.tab_index]
                coord['left'] += offset[0]
                coord['right'] += offset[0]
                coord['up'] += offset[1]
                coord['down'] += offset[1]
                visible_element['coord']['left'] = max(0, coord['left'])
                visible_element['coord']['right'] = min(1920, coord['right'])
                visible_element['coord']['up'] = max(0, coord['up'])
                visible_element['coord']['down'] = min(1080, coord['down'])

                visible_elements.append(visible_element)
        return visible_elements
    
    def scale_element_height(self, elements: list, page_scale=1.2, element_scale=1.5) -> list:
        # scale the height of the elements to be within 2 * height
        scale = page_scale * self.height / elements[-1]['coord']['down']
        # get random index from elements
        ind = random.randint(0, len(elements) - 1)
        for i, element in enumerate(elements):
            element['coord']['up'] = int(element['coord']['up'] * scale)
            element['coord']['down'] = int(element['coord']['down'] * scale)

        # expand the height of the elements a little bit
        displacement = int(elements[ind]['coord']['down'] * (element_scale - 1))
        for i, element in enumerate(elements):
            if i <= ind:
                element['coord']['up'] = int(element['coord']['up'] * element_scale)
                element['coord']['down'] = int(element['coord']['down'] * element_scale)
            else:
                element['coord']['up'] += displacement
                element['coord']['down'] += displacement

        return elements
        
    
    def remove_coord_info(self, elements: list) -> list:
        # elements: list of elements
        # remove the coord info from the elements in the tree
        
        for element in elements:
            if 'id' not in element['coord']:
                element['coord'] = {'id': random.randint(0, 20000)}
            else:
                element['coord'] = {'id': element['coord']['id']}
        return elements

    def erase_coord_info_in_tree(self, state: dict, duplicate = False) -> dict:
        if duplicate:
            tmp_state = copy.deepcopy(state)
        # clear the coordinate info in the tree

        def erase_coord(elements: list):
            for element in elements:
                if 'elements' in element:
                    erase_coord(element['elements'])
                if 'coord' in element:
                    element['coord'] = {'id': element['coord']['id']}
        if duplicate:
            erase_coord([tmp_state])
            return tmp_state
        else:
            erase_coord([state])
            return state
    
    def flatten_tree_state(self, state: dict, leaf_only = False) -> list[dict]:
        # flatten the tree structure of the state
        if isinstance(state, list):
            state = state[0]
        def flatten_tree(elements):
            flat_state = []
            for element in elements:
                if leaf_only:
                    if 'elements' in element:
                        flat_state.extend(flatten_tree(element['elements']))
                    else:
                        new_element = copy.deepcopy(element)
                        flat_state.append(new_element)
                else:
                    new_element = copy.deepcopy(element)
                    if 'elements' in element:
                        del new_element['elements']
                    flat_state.append(new_element)
                    if 'elements' in element:
                        flat_state.extend(flatten_tree(element['elements']))
            return flat_state
        if 'elements' in state:
            if leaf_only:
                return flatten_tree(state['elements'])
            else:
                new_element = copy.deepcopy(state)
                del new_element['elements']
                return [new_element] + flatten_tree(state['elements'])
    
    def transform_response_into_json(self, response: str):
        # simple cleaning
        response = response.strip()
        if response.startswith('```'):
            response = '\n'.join(response.split('\n')[1:-1])
        try:
            state = json.loads(response)
        except Exception as e:
            print(e)
            raise ValueError("Response is not a valid JSON format.")
        
        # state = self.collate_coord(state)
        state = self.shuffle_element_id(state)

        if self.webarena_mode:
            # 1. put the leaves nodes into one list
            elements = self.flatten_tree_state(state, leaf_only=True)

            # 2. render the leaves nodes
            elements = self._render_webarena_state(elements)

            # 3. scale the height of the new state to be within 1.5 * height
            elements = self.scale_element_height(elements, page_scale=1.0, element_scale=1.5)

            # 4. replace the coordinates with the rendered coordinates
            for element in elements:
                id = element['coord']['id']
                target_element = self.find_element_by_id(state, id)
                if target_element:
                    target_element['coord'] = element['coord']

        
        return state

    
    @timer
    def extract_intent(self, action):
        with open('system_prompts/web_simulation/intention_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        related_elements = []
        if isinstance(action, Click) or isinstance(action, Type):
            related_element = self.find_element_by_coordinates(self.cur_state, action.x, action.y, filter_static=True, filter_image=True)
            if not related_element:
                return "No related element found."
            elif isinstance(action, Type) and ('input' not in related_element['tag'] and 'textarea' not in related_element['tag']):
                return "No related element found."
            related_elements.append(related_element)

        elif isinstance(action, WA_Click) or isinstance(action, WA_Type) or isinstance(action, WA_Hover):

            related_element = self.find_element_by_id(self.cur_state, action.id, filter_static=True)
            if not related_element:
                return "No related element found."
            related_elements.append(related_element)
        
        prompt = '''GUI:
{}

Browsing history:
{}
Action definition: {}
Current action: {}
Related elements: 
{}\n**Note: try to use realistic URL, if needed. Don't use URL like "example.com".**'''.format(str(self.cur_state), '\n'.join(self.intent_history), action.INTRO, str(action), "\n".join([str(related_element) for related_element in related_elements]))
        response = self.call_openai(prompt, sys_prompt)

        if self.debug:
            self.logger.info(f"Extracted intent: {response}")
        return response
    
    
    @timer
    def direct_update(self, intent):
        prompt = """
Current state:
{}
Update Message: {}""".format(self._get_backup_current_state(), intent)
        path = 'system_prompts/web_simulation/direct_step.txt' if not self.webarena_mode else 'system_prompts/web_simulation/webarena_direct_step.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        response = self.call_openai(prompt, sys_prompt)
        response = self.transform_response_into_json(response)

        return response
    
    
    @timer
    def compose(self, intent, prev_state, polish = True):
        with open('system_prompts/web_simulation/compose_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = '''Previous Info: {}\n{}\nNote: Don't generate duplicate things! just put the elements and sections in the order that they shall appear in the webpage. E.g. you don't want to put footer before main content.\nThought: Let's think step by step. The description is'''.format('\n'.join(self.key_infos), intent.replace('New window', 'Descrition'))
        response = self.call_openai(prompt, sys_prompt, model='gpt-4o-mini')
        
        # further diversify the response
        if polish:
            with open('system_prompts/web_simulation/polish_compose.txt', 'r') as f:
                sys_prompt = f.read()
            prompt = '''Previous page:{}\nOld content:{}\n Thought: Let's think step by step. '''.format(prev_state, response)
            response = self.call_openai(prompt, sys_prompt, model='gpt-4o-mini', temperature=0.5)
            try:
                response = response.split("New content:")[1].strip()
            except:
                response = response
        if self.debug:
            self.logger.info(f"Composed state: {response}")
        return response
    
    
    @timer
    def format(self, composition, prev_state):
        path = 'system_prompts/web_simulation/format_prompt.txt' if not self.webarena_mode else 'system_prompts/web_simulation/webarena_format_prompt.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        prompt = f"""Previous state:\n{prev_state}\nDescription of new state: {composition}"""
        response = self.call_openai(prompt, sys_prompt, return_json=True)
        return response
    
    def complete_state(self, incomplete_state):
        old_elements = self.flatten_tree_state(incomplete_state, leaf_only=True)
        length = len(old_elements)
        with open('system_prompts/web_simulation/complete_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = f"""Incomplete state: {incomplete_state}"""
        response = self.call_openai(prompt, sys_prompt, return_json=True)

        # Tree-structured state
        state = json.loads(response)

        # start rendering the state
        new_elements = self.flatten_tree_state(state, leaf_only=True)
        new_elements = self._render_webarena_state(new_elements)

        # the scale factor: we need to ensure that old elements are within a screen
        scale = self.height / new_elements[length - 1]['coord']['down']
        for i, element in enumerate(new_elements):
            element['coord']['up'] = int(element['coord']['up'] * scale)
            element['coord']['down'] = int(element['coord']['down'] * scale)

        # replace the coordinates with the rendered coordinates
        for element in new_elements:
            id = element['coord']['id']
            target_element = self.find_element_by_id(state, id)
            if target_element:
                target_element['coord'] = element['coord']

        return state
    
    
    def create_new_page(self, cur_state, intent, polish=True):
        if polish:
            cur_state = self.erase_coord_info_in_tree(cur_state, duplicate=True)
            WA_cur_state = self.webarena_converter.convert_tree_venv_to_real_env(cur_state)
        else:
            WA_cur_state = None
        composition = self.compose(intent, WA_cur_state, polish=polish)
        formatted = self.format(composition, cur_state)
        
        # continue composition
        new_state = self.transform_response_into_json(formatted)
        return new_state
    
    def perturb_state(self, cur_state):
        with open('system_prompts/web_simulation/perturb_state.txt', 'r') as f:
            sys_prompt = f.read()
        
        prompt = f"""Current webpage: \n{cur_state}\n New webpage:"""

        response = self.call_openai(prompt, sys_prompt)

        return response

    def _get_backup_current_state(self):
        return self.prev_states[self.tab_index][self.window_index[self.tab_index]]
    
    def _render_webarena_state(self, state):
        path = 'system_prompts/web_simulation/webarena_render_prompt.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        prompt = str(state)
        response = self.call_openai(prompt, sys_prompt, return_json=True)
        
        rendered_elements = json.loads(response)
        return rendered_elements['elements']
    
    # Execute the action on current state
    # Return: whether need LLM to execute the action
    def execute_action(self, action) -> bool:
            
        if isinstance(action, Click) or isinstance(action, WA_Click):
            return True
        elif isinstance(action, Type) or isinstance(action, WA_Type):
            # find the element to type
            if isinstance(action, Type):
                typed_element = self.find_element_by_coordinates(self._get_backup_current_state(), action.x, action.y, filter_static=True, filter_image=True)
            else:
                
                typed_element = self.find_element_by_id(self._get_backup_current_state(), action.id, filter_static=True, filter_image=True)
                if not typed_element:
                    return False
                if action.press_enter_after == 1:
                    return True

            if typed_element:
                if ('input' in typed_element['tag'] or 'textarea' in typed_element['tag'] or 'combobox' in typed_element['tag']):
                    typed_element['content_description'] = action.text
                    return False
                elif 'select' in typed_element['tag']:
                    return False
            else:
                return True
            
        elif isinstance(action, Scroll) or isinstance(action, WA_Scroll):
            direction2offset = {'up': (0, 1080), 'down': (0, -1080), 'left': (200, 0), 'right': (-200, 0)}
            offset = direction2offset[action.direction]
            if action.direction == 'up' and self.coord_offset[self.tab_index][1] >= 0:
                return False
            cur_state = self._get_backup_current_state()
            leaf_node = self._get_leaf_node(cur_state)
            if 'left' not in leaf_node['coord']:
                return True
            self.coord_offset[self.tab_index] = (self.coord_offset[self.tab_index][0] + offset[0], self.coord_offset[self.tab_index][1] + offset[1])
            return False

        elif isinstance(action, WA_Navigate_Forward):
            if self.window_index[self.tab_index] < len(self.prev_states[self.tab_index]) - 1:
                self.window_index[self.tab_index] += 1
                
                self.coord_offset[self.tab_index] = (0, 0)
            return False

        elif isinstance(action, Navigate_Back) or isinstance(action, WA_Navigate_Back):
            if self.window_index[self.tab_index] > 0:
                self.window_index[self.tab_index] -= 1
                self.coord_offset[self.tab_index] = (0, 0)
            return False
        
        elif isinstance(action, Navigate_Home):
            self.prev_states[self.tab_index].append(self.home_state)
            self.window_index[self.tab_index] += 1
            self.coord_offset[self.tab_index] = (0, 0)
            return False
        
        elif isinstance(action, Wait):
            return False
        elif isinstance(action, WA_New_Tab):
            self.prev_states.append([self.home_state])
            self.window_index.append(0)

            self.tab_index = len(self.prev_states) - 1
            self.coord_offset.append((0, 0))
            return False
        
        elif isinstance(action, WA_Switch_Tab):
            self.tab_index = action.index
            return False
        elif isinstance(action, WA_Close_Tab):
            self.prev_states.pop(self.tab_index)
            self.window_index.pop(self.tab_index)
            self.coord_offset.pop(self.tab_index)
            self.tab_index -= 1
            return False
        
        elif isinstance(action, WA_Go_To):
            return True
        elif isinstance(action, WA_Press) or isinstance(action, WA_Hover):
            return True
        else:
            raise ValueError("Action not defined")

    def step(self, action):

        # execute the action
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------")
            print(intent)
            print("--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.cur_state
            
            # clean all empty str in the list
            thought, intent, key_info, answer = [item for item in intent.strip().split("\n") if item != '']
            
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                key_info = key_info.split('Key Info: ')[1].strip()
                self.key_infos.append(key_info)
            # judge whether creating a new page is required
            if "Yes" in answer:
                new_state = self.create_new_page(self._get_backup_current_state(), intent)

                if self.webarena_mode:
                    new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
                else:
                    new_cur_state = self.get_visible_elements(new_state)
                
                # update the simulator
                prev_coord_offset = self.coord_offset[self.tab_index]
                try:
                    # not using coord_offset in prev step
                    self.coord_offset[self.tab_index] = (0, 0)
                    if self.webarena_mode:
                        new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
                    else:
                        new_cur_state = self.get_visible_elements(new_state)

                    if isinstance(action, Type):
                        self.prev_states[self.tab_index][self.window_index[self.tab_index]] = new_state
                    else:
                        self.window_index[self.tab_index] += 1
                        self.prev_states[self.tab_index] = self.prev_states[self.tab_index][:self.window_index[self.tab_index]]
                        self.prev_states[self.tab_index].append(new_state)

                except:
                    self.coord_offset[self.tab_index] = prev_coord_offset
                    new_cur_state = self.remove_coord_info(self.get_visible_elements(self._get_backup_current_state()))

            else: 
                # directly update the current state
                new_state = self.direct_update(intent)

                if self.webarena_mode:
                    new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
                else:
                    new_cur_state = self.get_visible_elements(new_state)

                # update the simulator
                self.prev_states[self.tab_index][self.window_index[self.tab_index]] = new_state    
            self.intent_history.append(f"Step {self.steps + 1}: " + intent)
            

        else:
            new_cur_state = self.remove_coord_info(self.get_visible_elements(self._get_backup_current_state()))
            intent = f"Step {self.steps + 1}: " + str(action)
            self.intent_history.append(intent)
            
        # update the current state
        self.cur_state = new_cur_state
        self.steps += 1
        return self.cur_state

    def reset(self,
              init_state: dict,
              ):
        
        self.tab_index = 0
        self.window_index = [0]
        self.coord_offset = [(0, 0)]
        self.steps = 0

        # init_state: tree structure of the initial state
        leaf_node = self._get_leaf_node(init_state)
        if 'left' not in leaf_node['coord']:
            self.cur_state = self.flatten_tree_state(init_state)
        else:
            self.cur_state = self.remove_coord_info(self.get_visible_elements(init_state))
        self.prev_states = [[init_state]]

    def _get_state(self) -> dict:
        # function for getting intermediate state
        return {
            'cur_state': self.cur_state,
            'prev_states': self.prev_states,
            'tab_index': self.tab_index,
            'window_index': self.window_index,
            'coord_offset': self.coord_offset,
            'steps': self.steps,
            'intent_history': self.intent_history,
            'key_infos': self.key_infos
        }

    def _set_state(self, simulator_state: dict) -> None:
        # function for setting intermediate state
        self.cur_state = simulator_state['cur_state']
        self.prev_states = simulator_state['prev_states']
        self.tab_index = simulator_state['tab_index']
        self.window_index = simulator_state['window_index']

        self.coord_offset = simulator_state['coord_offset']
        self.steps = simulator_state['steps']
        self.intent_history = simulator_state['intent_history']
        self.key_infos = simulator_state['key_infos']


# Retrieval augmented simulator
class RAG_Simulator(Simulator):
    def __init__(self,
                 webarena_domain,
                 init_state=None,
                 width=1920,
                 height=1080,
                 webarena_mode=False,
                 debug=False,
                 ):
        super().__init__(init_state, width, height, webarena_mode, debug)
        self.webarena_domain = webarena_domain
        self.retrieve_key = None

    def reset(self, init_state: dict):
        super().reset(init_state)
        WA_init_state = self.webarena_converter.convert_tree_venv_to_real_env(init_state)
        self.retrieve_key = clean_state(WA_init_state)

    def model_retrieve(self, action_history, reference_seqs: list[str]):

        with open('system_prompts/web_simulation/model_retrieve.txt', 'r') as f:
            sys_prompt = f.read()
        reference_seqs = '\n'.join(reference_seqs)
        prompt = f"""Reference action sequences:\n{reference_seqs}\n\nCurrent action sequence:\n{action_history}\n\Thought:Let's think step by step."""
        response = self.call_openai(prompt, sys_prompt, model='gpt-4o', temperature=0.1)
        try:
            thought, output = response.split('Output:')
        except:
            output = response
        return output.strip().split('\n')

    def find_best_match(self, action_history: list[str]):
        # key: cleaned webarena state
        key = self.retrieve_key + ' '.join(action_history)
        root = f'intermediate_states/webarena/{self.webarena_domain}'
        files = [f for f in os.listdir(root) if f.endswith('.json')]

        def get_indexed_action_history(act_his: list[str]) -> str:
            return ' '.join([f"{i}. {act}" for i, act in enumerate(act_his)])
        
        keys = []
        actions = []
        action_lists = []
        for file in files:
            d = json.load(open(f'{root}/{file}', 'r'))
            keys.append(d['clean_prev_state'] + d['action_history'])
            actions.append(d['action_history'])
            if self.webarena_domain != 'map':
                action_lists.append(d['action_history_list'])

        # First BM25 over action_history to select top 10
        top_n_initial = max(len(keys) // 20, 1)  # Ensure at least one selection
        fuzzy, top_indices_initial_1st = BM25_ranking(' '.join(action_history), actions, top_n=top_n_initial)

        # filter out seqs that not ending with the same action as action_history
        if self.webarena_domain != 'map':
            tmp = [i for i in top_indices_initial_1st if action_lists[i][-1] == action_history[-1]]
            if tmp:
                top_indices_initial_1st = tmp

            filtered_actions = [get_indexed_action_history(action_lists[i]) for i in top_indices_initial_1st]
        else:
            filtered_actions = [actions[i] for i in top_indices_initial_1st]
        
        top_indices_initial_2nd = []
        if fuzzy:
            # do model-based retrieval
            # breakpoint()
            
            filtered_actions = self.model_retrieve(get_indexed_action_history(action_history), filtered_actions)
            if self.webarena_domain != 'map':
                top_indices_initial_2nd = [i for i in top_indices_initial_1st for a in filtered_actions if get_indexed_action_history(action_lists[i]) in a]
            else:
                top_indices_initial_2nd = [i for i in top_indices_initial_1st for a in filtered_actions if actions[i] in a]

        if not top_indices_initial_2nd:
            top_indices_initial = top_indices_initial_1st
        else:
            top_indices_initial = top_indices_initial_2nd
        filtered_files = [files[i] for i in top_indices_initial]
        filtered_keys = [keys[i] for i in top_indices_initial]
        
        # Then BM25 over retrieve_key on the filtered candidates
        _, top_indices_final = BM25_ranking(key, filtered_keys, top_n=1)
        best_match = json.load(open(f'{root}/{filtered_files[top_indices_final[0]]}', 'r'))
        return best_match
    
    @timer
    def rag_compose(self, intent, reference_state):
        path = 'system_prompts/web_simulation/rag_compose_prompt.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        prompt = f"""Reference state:\n{reference_state}\nDescription of new state: {intent}"""
        response = self.call_openai(prompt, sys_prompt)
        return response

    @timer
    def format(self, composition, reference_state):
        path = 'system_prompts/web_simulation/rag_format_prompt.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        prompt = f"""Reference state:\n{reference_state}\nDescription of new state: {composition}"""
        response = self.call_openai(prompt, sys_prompt, return_json=True)
        return response

    def create_new_page(self, WA_reference_state, intent):

        composition = self.rag_compose(intent, WA_reference_state)

        # breakpoint()
        reference_state = self.webarena_converter.convert_to_tree_venv(WA_reference_state)
        formatted = self.format(composition, reference_state)
        # breakpoint()
        new_state = self.transform_response_into_json(formatted)
        return new_state

    def step(self, action, action_history: list[str]):
        # execute the action
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------")
            print(intent)
            print("--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.cur_state
            
            # clean all empty str in the list
            thought, intent, key_info, answer = [item for item in intent.strip().split("\n") if item != '']
            
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                key_info = key_info.split('Key Info: ')[1].strip()
                self.key_infos.append(key_info)
            # judge whether creating a new page is required
            if "Yes" in answer:

                # here we need to retrieve the best matching state
                # used as reference state for generation
                best_match = self.find_best_match(action_history)
                new_state = self.create_new_page(best_match['next_state'], intent)

                if self.webarena_mode:
                    new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
                else:
                    new_cur_state = self.get_visible_elements(new_state)
                
                # update the simulator
                if isinstance(action, Type):
                    self.prev_states[self.tab_index][self.window_index[self.tab_index]] = new_state
                else:
                    self.window_index[self.tab_index] += 1
                    self.prev_states[self.tab_index] = self.prev_states[self.tab_index][:self.window_index[self.tab_index]]
                    self.prev_states[self.tab_index].append(new_state)
                    self.coord_offset[self.tab_index] = (0, 0)
                
                self.retrieve_key = best_match['clean_next_state']

            else: # directly update the current state
                if isinstance(action, Type) or isinstance(action, WA_Type):
                    tmp_action = Type(text=action.text, x=action.x, y=action.y, press_enter_after=0) if isinstance(action, Type) else WA_Type(id=action.id, text=action.text, press_enter_after=0)
                    self.execute_action(tmp_action)
                    if self.webarena_mode:
                        new_cur_state = self.remove_coord_info(self.get_visible_elements(self._get_backup_current_state()))
                    else:
                        new_cur_state = self.get_visible_elements(self._get_backup_current_state())
                else:
                    new_state = self.direct_update(intent)

                    if self.webarena_mode:
                        new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
                    else:
                        new_cur_state = self.get_visible_elements(new_state)

                    # update the simulator
                    self.prev_states[self.tab_index][self.window_index[self.tab_index]] = new_state    
            self.intent_history.append(f"Step {self.steps + 1}: " + str(action))

        else:
            new_cur_state = self.remove_coord_info(self.get_visible_elements(self._get_backup_current_state()))
            intent = f"Step {self.steps + 1}: " + str(action)
            self.intent_history.append(intent)
            
        # update the current state
        self.cur_state = new_cur_state
        self.steps += 1
        return self.cur_state

class AblationSimulator(Simulator):
    
    def step(self, action):
        # execute the action
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------")
            print(intent)
            print("--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.cur_state
            
            # clean all empty str in the list
            thought, intent, key_info, answer = [item for item in intent.strip().split("\n") if item != '']
            
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                key_info = key_info.split('Key Info: ')[1].strip()
                self.key_infos.append(key_info)
            
            new_state = self.direct_update(intent)

            if self.webarena_mode:
                new_cur_state = self.remove_coord_info(self.get_visible_elements(new_state))
            else:
                new_cur_state = self.get_visible_elements(new_state)

            # update the simulator
            if isinstance(action, Type):
                self.prev_states[self.tab_index][self.window_index[self.tab_index]] = new_state
            else:
                self.window_index[self.tab_index] += 1
                self.prev_states[self.tab_index] = self.prev_states[self.tab_index][:self.window_index[self.tab_index]]
                self.prev_states[self.tab_index].append(new_state)
            
            self.intent_history.append(f"Step {self.steps + 1}: " + str(action))

        else:
            new_cur_state = self.remove_coord_info(self.get_visible_elements(self._get_backup_current_state()))
            intent = f"Step {self.steps + 1}: " + str(action)
            self.intent_history.append(intent)
            
        # update the current state
        self.cur_state = new_cur_state
        self.steps += 1
        return self.cur_state

####################################
########## Android Simulator ##########
####################################

class UIElement:
    def __init__(self, 
                 text,
                 content_description,
                 class_name,
                 bbox,
                 bbox_pixels: tuple[int, int, int, int],
                 hint_text = None,
                 is_checked=False,
                 is_checkable=False,
                 is_clickable=False,
                 is_editable=False,
                 is_enabled=False,
                 is_focused=False,
                 is_focusable=False,
                 is_long_clickable=False,
                 is_scrollable=False,
                 is_selected=False,
                 is_visible=False,
                 package_name='',
                 resource_name='',
                 tooltip=None,
                 resource_id=None,
                 metadata=None,
                 ):
        self.text = text
        self.content_description = content_description
        self.class_name = class_name
        self.bbox = bbox
        self.bbox_pixels = bbox_pixels
        self.hint_text = hint_text
        self.is_checked = is_checked
        self.is_checkable = is_checkable
        self.is_clickable = is_clickable
        self.is_editable = is_editable
        self.is_enabled = is_enabled
        self.is_focused = is_focused
        self.is_focusable = is_focusable
        self.is_long_clickable = is_long_clickable
        self.is_scrollable = is_scrollable
        self.is_selected = is_selected
        self.is_visible = is_visible
        self.package_name = package_name
        self.resource_name = resource_name
        self.tooltip = tooltip
        self.resource_id = resource_id
        self.metadata = metadata

    def to_str(self):
        bounding_box = f"BoundingBox(x_min={self.bbox_pixels[0]}, x_max={self.bbox_pixels[1]}, y_min={self.bbox_pixels[2]}, y_max={self.bbox_pixels[3]})"
        return f"UIElement(text={self.text}, content_description={self.content_description}, class_name={self.class_name}, bbox={self.bbox}, bbox_pixels={bounding_box}, hint_text={self.hint_text}, is_checked={self.is_checked}, is_clickable={self.is_clickable}, is_editable={self.is_editable}, is_enabled={self.is_enabled}, is_focused={self.is_focused}, is_focusable={self.is_focusable}, is_long_clickable={self.is_long_clickable}, is_scrollable={self.is_scrollable}, is_selected={self.is_selected}, is_visible={self.is_visible}, package_name={self.package_name}, resource_name={self.resource_name}, tooltip={self.tooltip}, resource_id={self.resource_id}, metadata={self.metadata})"
    
    def __str__(self):
        return self.to_str()

class AndroidSimulator:
    def __init__(self,
                 init_state=None,
                 width=1080,
                 height=2400,
                 debug = False,
                 ):
        self.client = OpenAI(
            organization=os.environ['OPENAI_ORG_ID'],
            api_key=os.environ['OPENAI_API_KEY']
        )
        self.init_state = init_state

        # state in AndroidWorld: list of UIElements
        self.cur_state = []
        self.home_state = []
        self.steps = 0

        self.prev_states = []
        self.window_index = -1
        self.coord_offset = []

        self.intent_history = []
        self.key_infos = []

        # screen parameters
        self.width = width
        self.height = height

        # logging module
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.INFO,
                                filemode='w',
                                filename='tmp/simulator.log',
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

    def call_openai(self,
                    prompt, 
                    sys_prompt, 
                    model="gpt-4o-mini",
                    stop=None, 
                    return_json=False,
                    max_tokens=None,
                    temperature=0.5):
        completion = self.client.chat.completions.create(
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
        caller = inspect.stack()[1].function
        usage = completion.usage
        print(f"Function ##{caller}## called with usage: Input token:{usage.prompt_tokens}, Output token:{usage.completion_tokens}")
        # print(completion)
        # print(type(completion.choices[0].message.content))
        return completion.choices[0].message.content

    
    # Function to locate an element by coordinates
    @staticmethod
    def find_element_by_coordinates(elements: list[UIElement], x, y):
        for element in elements:
            bbox_pixels = element.bbox_pixels
            if bbox_pixels[0] <= x <= bbox_pixels[1] and bbox_pixels[2] <= y <= bbox_pixels[3]:
                return element
                
        return None
    
    def find_element_by_id(self, elements, id):
        return elements[int(id) - 1]

    @timer
    def extract_intent(self, action):
        with open('system_prompts/android_simulation/intention_prompt.txt', 'r') as f:
            sys_prompt = f.read()

        related_element = None
        if isinstance(action, Click) or isinstance(action, Type):
            related_element = self.cur_state[int(action.id)]
        
        prompt = '''GUI:
{}

Action history:
{}
Action definition: {}
Current action: {}
Related element: {}\n**Note: try to use realistic content, if needed. Don't use content like "example.com".**'''.format(str(self.cur_state), '\n'.join(self.intent_history), action.INTRO, str(action), str(related_element))
        # print(related_elements)
        response = self.call_openai(prompt, sys_prompt)

        if self.debug:
            self.logger.info(f"Extracted intent: {response}")
        return response
    
    @timer
    def direct_update(self, intent):
        prompt = """
Current state:
{}
Update Message: {}""".format(self._get_backup_current_state(), intent)
        path = 'system_prompts/android_simulation/direct_step.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        response = self.call_openai(prompt, sys_prompt)
        if response.startswith('```'):
            # remove the first and the last line
            response = response.strip().split('\n')[1:-1]
            response = '\n'.join(response)

        return response
    
    @timer
    def compose(self, intent, prev_state):
        with open('system_prompts/android_simulation/compose_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = '''Previous Info: {}\n{}\nThought: Let's think step by step. The description is'''.format('\n'.join(self.key_infos), intent.replace('New window', 'Descrition'))
        response = self.call_openai(prompt, sys_prompt, model='gpt-4o-mini')
        
        if self.debug:
            self.logger.info(f"Composed state: {response}")
        return response
    
    @timer
    def format(self, composition, prev_state):
        path = 'system_prompts/android_simulation/format_prompt.txt'
        with open(path, 'r') as f:
            sys_prompt = f.read()
        prompt = f"""Previous state:\n{prev_state}\nDescription of new state: {composition}"""
        response = self.call_openai(prompt, sys_prompt)
        print("Format response: ", response)
        if response.startswith('```'):
            # remove the first and the last line
            response = response.strip().split('\n')[1:-1]
            response = '\n'.join(response)
        return response
    
    def create_new_page(self, cur_state, intent):
        composition = self.compose(intent, cur_state)
        formatted = self.format(composition, cur_state)
        return formatted
    
    def perturb_state(self, cur_state):
        with open('system_prompts/android_simulation/perturb_state.txt', 'r') as f:
            sys_prompt = f.read()

        # 1. only keeping the text and content_description from the list of UI elements
        parsed_cur_state = [self.parse_ui_element_string(s) for s in cur_state.split('\n')]
        shortened_cur_state = self.get_valuable_short_state(cur_state)
        prompt = f"""Current UI: \n{shortened_cur_state}\n New UI:"""

        response = self.call_openai(prompt, sys_prompt)

        # 2. return the value to the parsed_cur_state
        new_state = ''
        for line, element in zip(response.split('\n'), parsed_cur_state):
            if line.startswith('UIElement('):
                new_element = self.parse_ui_element_string(line)
                element.text = new_element.text
                element.content_description = new_element.content_description

                new_state += element.to_str() + '\n'
        return new_state

    def _get_backup_current_state(self):
        return self.prev_states[self.window_index]
    
    def get_visible_elements(self, elements):
        visible_elements = []
        for element in elements:
            if element.class_name == 'android.view.View':
                visible_elements.append(element)
            elif element.package_name == 'com.android.systemui':
                visible_elements.append(element)
            else:
                offset = self.coord_offset[self.window_index]
                if element.bbox_pixels[0] + offset[0] >= 0 and element.bbox_pixels[1] + offset[0] <= self.width and element.bbox_pixels[2] + offset[1] >= 0 and element.bbox_pixels[3] + offset[1] <= self.height:
                    element.is_visible = True
                    visible_elements.append(element)
        return visible_elements

    
    # Execute the action on current state
    # Return: whether need LLM to execute the action
    def execute_action(self, action) -> bool:
        if isinstance(action, Click) or isinstance(action, OpenApp):
            return True
        
        elif isinstance(action, Scroll):
            # change the interface by modifying the coordinates and finally get the visible elements.
            direction2offset = {'up': (0, 1200), 'down': (0, -1200), 'left': (200, 0), 'right': (-200, 0)}
            offset = direction2offset[action.direction]
            if action.direction == 'up' and self.coord_offset[self.window_index][1] >= 0:
                return False
            self.coord_offset[self.window_index] = (self.coord_offset[self.window_index][0] + offset[0], self.coord_offset[self.window_index][1] + offset[1])
            return False
        elif isinstance(action, Type):
            typed_element = self.cur_state[int(action.id)]
            if typed_element:
                if typed_element.text is None:
                    typed_element.text = action.text
                else:
                    typed_element.content_description = action.text
                return False
            else:
                return True
        elif isinstance(action, Navigate_Back):
            if self.window_index > 0:
                self.window_index -= 1
            return False
        elif isinstance(action, Navigate_Home):
            self.prev_states = [self.home_state]
            self.window_index = 0
            return False
    
        elif isinstance(action, Wait):
            return False
        elif isinstance(action, KeyboradEnter):
            return True

    def step(self, action):
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------")
            print(intent)
            print("--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.get_str_cur_state()
            
            # clean all empty str in the list
            thought, intent, key_info, answer = [item for item in intent.strip().split("\n") if item != '']
            
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                key_info = key_info.split('Key Info: ')[1].strip()
                self.key_infos.append(key_info)
            # judge whether creating a new page is required
            if "Yes" in answer:
                new_state = self.create_new_page(self._get_backup_current_state(), intent)
                new_state = [self.parse_ui_element_string(s) for s in new_state.split('\n')]

                new_cur_state = self.get_visible_elements(new_state)

                # update the simulator
                self.window_index += 1
                self.prev_states = self.prev_states[:self.window_index]
                self.prev_states.append(new_state)
                self.coord_offset.append((0, 0))

            else:
                new_state = self.direct_update(intent)
                new_state = [self.parse_ui_element_string(s) for s in new_state.split('\n')]
                new_cur_state = self.get_visible_elements(new_state)

                # update the simulator
                self.prev_states[self.window_index] = new_state
            self.intent_history.append(f"Step {self.steps + 1}: " + intent)
        else:
            new_cur_state = self.get_visible_elements(self._get_backup_current_state())
            intent = f"Step {self.steps + 1}: " + str(action)
            self.intent_history.append(intent)
        # update the current state
        self.cur_state = new_cur_state
        self.steps += 1
        return self.get_str_cur_state()
        

    def parse_bounding_box(self, s: str) -> Optional[tuple]:
        match = re.search(r'BoundingBox\(x_min=(\d+), x_max=(\d+), y_min=(\d+), y_max=(\d+)\)', s)
        if match:
            return tuple(map(int, match.groups()))
        return None

    def parse_ui_element_string(self, s: str) -> UIElement:
        bbox_pixels = self.parse_bounding_box(s)

        # Remove the BoundingBox(...) so eval can parse the rest
        s_cleaned = re.sub(r'bbox_pixels=BoundingBox\(.*?\)', 'bbox_pixels=None', s)

        # Convert to a dictionary-like string
        s_cleaned = s_cleaned.strip()
        s_cleaned = s_cleaned.removeprefix('UIElement(').removesuffix(')')
        
        # Create a dict by splitting key=value pairs
        field_dict = {}
        for kv in re.split(r', (?=\w+=)', s_cleaned):
            if not kv:
                continue
            try:
                key, value = kv.split('=', 1)
            except:
                breakpoint()
            key = key.strip()
            value = value.strip()
            if value == 'None':
                field_dict[key] = None
            elif value == 'True':
                field_dict[key] = True
            elif value == 'False':
                field_dict[key] = False
            elif value.startswith("'") and value.endswith("'"):
                field_dict[key] = value.strip("'")
            elif value.startswith('"') and value.endswith('"'):
                field_dict[key] = value.strip('"')
            else:
                field_dict[key] = value  # Assume string for now

        

        # Add parsed bbox_pixels
        field_dict['bbox_pixels'] = bbox_pixels

        # Provide missing optional values if not found
        all_fields = {
            'text': None, 'content_description': None, 'class_name': '',
            'bbox': None, 'bbox_pixels': None, 'hint_text': None,
            'is_checked': False, 'is_checkable': False, 'is_clickable': False,
            'is_editable': False, 'is_enabled': False, 'is_focused': False,
            'is_focusable': False, 'is_long_clickable': False, 'is_scrollable': False,
            'is_selected': False, 'is_visible': False, 'package_name': '',
            'resource_name': '', 'tooltip': None, 'resource_id': None, 'metadata': None
        }

        for k in all_fields:
            if k not in field_dict:
                field_dict[k] = all_fields[k]

        return UIElement(**field_dict)

    def get_str_cur_state(self):
        # convert the current state to a string
        cur_state_str = '\n'.join([str(element) for element in self.cur_state])
        return cur_state_str
    
    def get_str_backup_current_state(self):
        # convert the current state to a string
        cur_state_str = '\n'.join([str(element) for element in self._get_backup_current_state()])
        return cur_state_str

    def reset(self, init_state: str):
        self.window_index = 0
        self.coord_offset = [(0, 0)]
        parsed_init_state = [self.parse_ui_element_string(s) for s in init_state.split('\n')]
        self.cur_state = self.get_visible_elements(parsed_init_state)
        self.prev_states = [parsed_init_state]
        self.intent_history = []
        self.key_infos = []
        self.steps = 0
        

        return

class RAG_AndroidSimulator(AndroidSimulator):
    def __init__(self,
                 domain: str,
                 init_state=None,
                 width=1080,
                 height=2400,
                 debug=False):
        super().__init__(init_state, width, height, debug)
        # domain: APP name
        self.domain = domain
        self.retrieve_key = None

    def reset(self, init_state: str):
        super().reset(init_state)
        self.retrieve_key = init_state  # could add more processing

    def model_retrieve(self, action_history, reference_seqs: list[str]):
        with open('system_prompts/android_simulation/model_retrieve.txt', 'r') as f:
            sys_prompt = f.read()
        reference_seqs_str = '\n'.join(reference_seqs)
        prompt = f"""Reference actions:\n{reference_seqs_str}\n\nCurrent action sequence:\n{action_history}\nThought: Let's think step by step."""
        response = self.call_openai(prompt, sys_prompt, model='gpt-4o', temperature=0.1)
        try:
            _, output = response.split('Output:')
        except:
            output = response
        return output.strip().split('\n')

    def find_best_match(self, action_history: list[str]):
        key = self.retrieve_key + ' '.join(action_history)
        root = f'intermediate_states/android/{self.domain}'
        files = [f for f in os.listdir(root) if f.endswith('.json')]

        def get_indexed_action_history(act_his: list[str]) -> str:
            return ' '.join([f"{i}. {act}" for i, act in enumerate(act_his)])
        
        keys = []
        actions = []
        action_lists = []
        for file in files:
            d = json.load(open(f'{root}/{file}', 'r'))
            keys.append(d['prev_elements'][0] + ' '.join(d['action_history']))
            actions.append(' '.join(d['action_history']))
            action_lists.append(d['action_history'])

        top_n_initial = max(len(keys) // 5, 1)
        fuzzy, top_indices = BM25_ranking(key, keys, top_n=top_n_initial)
        tmp = [i for i in top_indices if action_lists[i][-1] == action_history[-1]]
        if tmp:
            top_indices = tmp
        filtered_actions = [get_indexed_action_history(action_lists[i]) for i in top_indices]
        top_indices_final = []

        # if fuzzy:
        filtered_actions = self.model_retrieve(get_indexed_action_history(action_history), filtered_actions)
        if filtered_actions == 'None':
            return {
                'next_elements': ['None'],
                'prev_elements': ['None'],
                'action_history': ['None']
            }
        top_indices_final = [i for i in top_indices for a in filtered_actions if get_indexed_action_history(action_lists[i]) in a]

        final_indices = top_indices_final if top_indices_final else top_indices
        best_file = files[final_indices[0]]
        return json.load(open(f'{root}/{best_file}', 'r'))

    @timer
    def rag_compose(self, intent, reference_state: str):
        with open('system_prompts/android_simulation/rag_compose_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = f"Reference state:\n{reference_state}\nDescription of new state: {intent}"
        return self.call_openai(prompt, sys_prompt, model='gpt-4o-mini')

    @timer
    def format(self, composition, reference_state: str):
        with open('system_prompts/android_simulation/rag_format_prompt.txt', 'r') as f:
            sys_prompt = f.read()
        prompt = f"Reference state:\n{reference_state}\nDescription of new state: {composition}"
        return self.call_openai(prompt, sys_prompt, model='gpt-4o-mini')

    def create_new_page(self, reference_state: str, intent: str):
        composition = self.rag_compose(intent, reference_state)
        return composition

    def step(self, action, action_history: list[str]):
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------\n", intent, "\n--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.get_str_cur_state()

            thought, intent, key_info, answer = [i for i in intent.strip().split('\n') if i]
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                self.key_infos.append(key_info.split('Key Info: ')[1].strip())

            if "Yes" in answer:
                best_match = self.find_best_match(action_history)
                new_state_str = self.create_new_page(best_match['next_elements'][0], intent)
                new_state = [self.parse_ui_element_string(s) for s in new_state_str.split('\n')]
                new_cur_state = self.get_visible_elements(new_state)

                self.window_index += 1
                self.prev_states = self.prev_states[:self.window_index]
                self.prev_states.append(new_state)
                self.coord_offset.append((0, 0))
                self.retrieve_key = best_match['next_elements'][0]
            else:
                new_state_str = self.direct_update(intent)
                new_state = [self.parse_ui_element_string(s) for s in new_state_str.split('\n')]
                new_cur_state = self.get_visible_elements(new_state)
                self.prev_states[self.window_index] = new_state

            self.intent_history.append(f"Step {self.steps + 1}: " + str(action))
        else:
            new_cur_state = self.get_visible_elements(self._get_backup_current_state())
            self.intent_history.append(f"Step {self.steps + 1}: " + str(action))

        self.cur_state = new_cur_state
        self.steps += 1
        return self.get_str_cur_state()

class AblationAndroidSimulator(AndroidSimulator):
    def step(self, action):
        if self.execute_action(action):
            intent = self.extract_intent(action)
            print("--------------")
            print(intent)
            print("--------------")
            if "No related element found" in intent:
                self.intent_history.append(f"Step {self.steps + 1}: Nothing happened.")
                self.steps += 1
                return self.get_str_cur_state()
            
            thought, intent, key_info, answer = [item for item in intent.strip().split("\n") if item != '']
            
            thought = thought.split(': ')[1].strip()
            if 'None' not in key_info:
                key_info = key_info.split('Key Info: ')[1].strip()
                self.key_infos.append(key_info)
            
            new_state_str = self.direct_update(intent)
            new_state = [self.parse_ui_element_string(s) for s in new_state_str.split('\n')]
            new_cur_state = self.get_visible_elements(new_state)

            # update the simulator
            self.window_index += 1
            self.prev_states = self.prev_states[:self.window_index]
            self.prev_states.append(new_state)
            self.coord_offset.append((0, 0))
            
            self.intent_history.append(f"Step {self.steps + 1}: " + str(action))

        else:
            new_cur_state = self.get_visible_elements(self._get_backup_current_state())
            intent = f"Step {self.steps + 1}: " + str(action)
            self.intent_history.append(intent)
            
        # update the current state
        self.cur_state = new_cur_state
        self.steps += 1
        return self.get_str_cur_state()